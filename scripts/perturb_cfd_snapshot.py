#!/usr/bin/env python3
"""Generate deterministic Xiao-style random perturbations of CFD states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cantera as ct
import h5py
import numpy as np


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _heat_release_rates(
    states: np.ndarray,
    mechanism: Path,
    phase_name: str,
) -> np.ndarray:
    gas = ct.Solution(str(mechanism), phase_name or None)
    rates = np.empty(states.shape[0], dtype=np.float64)
    for index, state in enumerate(states):
        gas.TPY = float(state[0]), float(state[1]), state[2:]
        rates[index] = float(gas.heat_release_rate)
    return rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phase-name", default="")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--temperature-amplitude-k", type=float, default=100.0)
    parser.add_argument("--pressure-span-fraction", type=float, default=0.15)
    parser.add_argument("--species-exponent-amplitude", type=float, default=0.15)
    parser.add_argument("--minimum-temperature-k", type=float, default=290.0)
    parser.add_argument("--nitrogen-span-fraction", type=float, default=0.05)
    parser.add_argument(
        "--negative-heat-release-fraction", type=float, default=0.05
    )
    parser.add_argument("--heat-release-floor", type=float, default=1.0)
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()

    if args.rounds < 1:
        raise ValueError("--rounds must be positive")

    with h5py.File(args.reference, "r") as handle:
        reference_states = np.asarray(handle["cells/states"], dtype=np.float64)
        reference_species = _decode(np.asarray(handle["species_names"]))

    with h5py.File(args.source, "r") as handle:
        source_states = np.asarray(handle["cells/states"], dtype=np.float64)
        species_names = _decode(np.asarray(handle["species_names"]))
        source_indices = np.asarray(
            handle["cells/source_cell_index"], dtype=np.int64
        )
        coordinates = np.asarray(handle["cells/coordinates"], dtype=np.float64)

    if species_names != reference_species:
        raise ValueError("source and reference species ordering differs")
    if source_states.shape[1] != len(species_names) + 2:
        raise ValueError("states must contain T, P, and all species")

    nitrogen_index = species_names.index("N2")
    pressure_span = float(np.ptp(reference_states[:, 1]))
    maximum_temperature = float(np.max(reference_states[:, 0]))
    nitrogen = reference_states[:, 2 + nitrogen_index]
    nitrogen_span = float(np.ptp(nitrogen))
    nitrogen_minimum = float(np.min(nitrogen)) - (
        args.nitrogen_span_fraction * nitrogen_span
    )
    nitrogen_maximum = float(np.max(nitrogen)) + (
        args.nitrogen_span_fraction * nitrogen_span
    )

    parent_heat_release = _heat_release_rates(
        source_states, args.mechanism, args.phase_name
    )
    accepted_states: list[np.ndarray] = []
    accepted_coordinates: list[np.ndarray] = []
    accepted_source_indices: list[np.ndarray] = []
    accepted_parent_indices: list[np.ndarray] = []
    accepted_rounds: list[np.ndarray] = []
    accepted_original: list[np.ndarray] = []
    accepted_heat_release: list[np.ndarray] = []

    if args.include_original:
        count = source_states.shape[0]
        accepted_states.append(source_states)
        accepted_coordinates.append(coordinates)
        accepted_source_indices.append(source_indices)
        accepted_parent_indices.append(source_indices)
        accepted_rounds.append(np.full(count, -1, dtype=np.int32))
        accepted_original.append(np.ones(count, dtype=np.uint8))
        accepted_heat_release.append(parent_heat_release)

    generated_count = 0
    rejected_temperature = 0
    rejected_pressure = 0
    rejected_nitrogen = 0
    rejected_heat_release = 0

    for round_index in range(args.rounds):
        seed_sequence = np.random.SeedSequence(
            [args.seed, args.shard_index, round_index]
        )
        rng = np.random.default_rng(seed_sequence)
        sample_count = source_states.shape[0]
        generated_count += sample_count

        perturbed = source_states.copy()
        perturbed[:, 0] += args.temperature_amplitude_k * rng.uniform(
            -1.0, 1.0, size=sample_count
        )
        perturbed[:, 1] += (
            args.pressure_span_fraction
            * pressure_span
            * rng.uniform(-1.0, 1.0, size=sample_count)
        )

        exponent = 1.0 + args.species_exponent_amplitude * rng.uniform(
            -1.0, 1.0, size=(sample_count, len(species_names))
        )
        with np.errstate(invalid="ignore", over="ignore", under="ignore"):
            perturbed_y = np.power(source_states[:, 2:], exponent)
        mass_sum = np.sum(perturbed_y, axis=1, keepdims=True)
        perturbed_y = np.divide(
            perturbed_y,
            mass_sum,
            out=np.zeros_like(perturbed_y),
            where=mass_sum > 0.0,
        )
        perturbed[:, 2:] = perturbed_y

        valid_temperature = (
            (perturbed[:, 0] >= args.minimum_temperature_k)
            & (perturbed[:, 0] <= maximum_temperature + args.temperature_amplitude_k)
            & np.isfinite(perturbed[:, 0])
        )
        valid_pressure = (perturbed[:, 1] > 0.0) & np.isfinite(perturbed[:, 1])
        perturbed_nitrogen = perturbed[:, 2 + nitrogen_index]
        valid_nitrogen = (
            (perturbed_nitrogen >= nitrogen_minimum)
            & (perturbed_nitrogen <= nitrogen_maximum)
        )
        valid_composition = (
            np.all(np.isfinite(perturbed_y), axis=1)
            & np.all(perturbed_y >= 0.0, axis=1)
            & (np.abs(np.sum(perturbed_y, axis=1) - 1.0) <= 1.0e-12)
        )
        prefilter = (
            valid_temperature
            & valid_pressure
            & valid_nitrogen
            & valid_composition
        )

        candidate_indices = np.flatnonzero(prefilter)
        candidate_heat_release = _heat_release_rates(
            perturbed[candidate_indices], args.mechanism, args.phase_name
        )
        parent_scale = np.maximum(
            np.abs(parent_heat_release[candidate_indices]),
            args.heat_release_floor,
        )
        valid_heat_release = candidate_heat_release >= (
            -args.negative_heat_release_fraction * parent_scale
        )
        accepted_indices = candidate_indices[valid_heat_release]

        rejected_temperature += int(np.count_nonzero(~valid_temperature))
        rejected_pressure += int(np.count_nonzero(valid_temperature & ~valid_pressure))
        rejected_nitrogen += int(
            np.count_nonzero(valid_temperature & valid_pressure & ~valid_nitrogen)
        )
        rejected_heat_release += int(np.count_nonzero(~valid_heat_release))

        accepted_states.append(perturbed[accepted_indices])
        accepted_coordinates.append(coordinates[accepted_indices])
        accepted_source_indices.append(source_indices[accepted_indices])
        accepted_parent_indices.append(source_indices[accepted_indices])
        accepted_rounds.append(
            np.full(accepted_indices.size, round_index, dtype=np.int32)
        )
        accepted_original.append(
            np.zeros(accepted_indices.size, dtype=np.uint8)
        )
        accepted_heat_release.append(candidate_heat_release[valid_heat_release])

    states = np.concatenate(accepted_states, axis=0)
    output_coordinates = np.concatenate(accepted_coordinates, axis=0)
    output_source_indices = np.concatenate(accepted_source_indices, axis=0)
    output_parent_indices = np.concatenate(accepted_parent_indices, axis=0)
    perturbation_round = np.concatenate(accepted_rounds, axis=0)
    is_original = np.concatenate(accepted_original, axis=0)
    heat_release = np.concatenate(accepted_heat_release, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype("utf-8")
    with h5py.File(args.output, "w") as handle:
        handle.attrs["schema_version"] = "cfd-cell-snapshot-v1"
        handle.attrs["source_kind"] = "xiao-random-perturbation"
        handle.attrs["source_snapshot"] = str(args.source)
        handle.attrs["reference_snapshot"] = str(args.reference)
        handle.attrs["sample_count"] = states.shape[0]
        handle.attrs["seed"] = args.seed
        handle.attrs["perturbation_rounds"] = args.rounds
        handle.attrs["config"] = json.dumps(vars(args), default=str, sort_keys=True)
        cells = handle.create_group("cells")
        cells.create_dataset("states", data=states, compression="gzip")
        cells.create_dataset(
            "coordinates", data=output_coordinates, compression="gzip"
        )
        cells.create_dataset(
            "source_cell_index", data=output_source_indices, compression="gzip"
        )
        cells.create_dataset(
            "parent_source_cell_index",
            data=output_parent_indices,
            compression="gzip",
        )
        cells.create_dataset(
            "perturbation_round", data=perturbation_round, compression="gzip"
        )
        cells.create_dataset(
            "is_original", data=is_original, compression="gzip"
        )
        cells.create_dataset(
            "reaction_rate", data=heat_release, compression="gzip"
        )
        handle.create_dataset(
            "species_names",
            data=np.asarray(species_names, dtype=object),
            dtype=string_dtype,
        )

    summary = {
        "source": str(args.source),
        "reference": str(args.reference),
        "output": str(args.output),
        "source_states": int(source_states.shape[0]),
        "rounds": args.rounds,
        "generated_perturbations": generated_count,
        "retained_originals": (
            int(source_states.shape[0]) if args.include_original else 0
        ),
        "accepted_states": int(states.shape[0]),
        "accepted_perturbations": int(np.count_nonzero(is_original == 0)),
        "perturbation_acceptance_fraction": float(
            np.count_nonzero(is_original == 0) / max(generated_count, 1)
        ),
        "rejected_temperature": rejected_temperature,
        "rejected_pressure": rejected_pressure,
        "rejected_nitrogen": rejected_nitrogen,
        "rejected_heat_release": rejected_heat_release,
        "heat_release_min": float(np.min(heat_release)),
        "heat_release_max": float(np.max(heat_release)),
        "temperature_min_k": float(np.min(states[:, 0])),
        "temperature_max_k": float(np.max(states[:, 0])),
        "maximum_mass_sum_error": float(
            np.max(np.abs(np.sum(states[:, 2:], axis=1) - 1.0))
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
