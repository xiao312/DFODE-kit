#!/usr/bin/env python3
"""Evaluate an exported Fluent artifact on untouched CFD interval pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import cantera as ct

from dfode_kit.data.cfd_conditioned import canonical_species_name


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _split_model_output(
    model_output: np.ndarray,
    n_species: int,
    manifest: dict,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    closure = manifest.get("thermochemical_closure", {})
    predicts_total_enthalpy = bool(
        closure.get("predicts_total_enthalpy_increment", False)
    )
    predicts_temperature = bool(closure.get("predicts_temperature", False))
    if predicts_total_enthalpy and predicts_temperature:
        raise ValueError("artifact cannot predict both temperature and total enthalpy")
    if model_output.shape[1] == n_species:
        if predicts_total_enthalpy or predicts_temperature:
            raise ValueError("artifact manifest requires a missing thermochemical output")
        return None, None, model_output
    if model_output.shape[1] != n_species + 1:
        raise ValueError(
            "Artifact output width does not match species or "
            f"thermochemical-plus-species contract: {model_output.shape[1]}"
        )
    scalar = model_output[:, 0]
    species = model_output[:, 1:]
    if predicts_total_enthalpy:
        return None, scalar, species
    if predicts_temperature or not closure:
        return scalar, None, species
    raise ValueError("artifact thermochemical manifest does not identify output scalar")


def _safe_cosine(predicted: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(predicted.ravel(), target.ravel()) / denominator)


def _subset_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    row_cosine: np.ndarray,
    mask: np.ndarray,
) -> dict:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"count": 0}
    error = predicted[mask] - target[mask]
    absolute_target = float(np.sum(np.abs(target[mask])))
    cosine_values = row_cosine[mask]
    return {
        "count": count,
        "fraction": float(np.mean(mask)),
        "delta_mae": float(np.mean(np.abs(error))),
        "normalized_absolute_error": float(
            np.sum(np.abs(error)) / max(absolute_target, 1.0e-300)
        ),
        "global_cosine": _safe_cosine(predicted[mask], target[mask]),
        "median_row_cosine": float(np.nanmedian(cosine_values)),
        "p10_row_cosine": float(np.nanpercentile(cosine_values, 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reactive-threshold", type=float, default=1.0e-8)
    parser.add_argument("--sign-threshold", type=float, default=1.0e-12)
    parser.add_argument("--max-normalized-mae", type=float, default=1.0)
    parser.add_argument("--minimum-global-cosine", type=float, default=0.5)
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-15)
    parser.add_argument(
        "--max-temperature-delta-nmae", type=float, default=float("inf")
    )
    parser.add_argument("--minimum-temperature", type=float, default=1.0)
    parser.add_argument("--maximum-temperature", type=float, default=5000.0)
    parser.add_argument(
        "--max-total-enthalpy-delta-nmae", type=float, default=float("inf")
    )
    parser.add_argument("--max-source-nmae", type=float, default=float("inf"))
    parser.add_argument(
        "--max-mixture-molecular-weight-nmae", type=float, default=float("inf")
    )
    parser.add_argument("--max-mass-sum-error", type=float, default=1.0e-12)
    parser.add_argument("--max-element-residual", type=float, default=1.0e-12)
    args = parser.parse_args()

    with h5py.File(args.source, "r") as handle:
        if "pairs" in handle:
            pairs = handle["pairs"]
            current_key = "current_states"
            target_key = "target_states"
        else:
            pairs = handle
            current_key = "state_before"
            target_key = "state_after"
        current = np.asarray(pairs[current_key], dtype=np.float64)
        target = np.asarray(pairs[target_key], dtype=np.float64)
        dt = np.asarray(pairs["dt"], dtype=np.float64)
        species_names = _decode(np.asarray(handle["species_names"]))
        source_index = (
            np.asarray(pairs["source_index"], dtype=np.int64)
            if "source_index" in pairs
            else np.arange(current.shape[0], dtype=np.int64)
        )
        h_total_before = (
            np.asarray(pairs["h_total_before"], dtype=np.float64)
            if "h_total_before" in pairs
            else None
        )
        h_total_after = (
            np.asarray(pairs["h_total_after"], dtype=np.float64)
            if "h_total_after" in pairs
            else None
        )

    model = torch.jit.load(
        str(args.artifact_dir / "model.pt"), map_location=args.device
    )
    model.eval()
    manifest = json.loads(
        (args.artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    separate_species_input = (
        manifest.get("hard_layer", {}).get("current_species_source")
        == "separate_input"
    )
    predicted_delta_parts = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, current.shape[0], args.batch_size):
            end = min(begin + args.batch_size, current.shape[0])
            model_input = np.concatenate(
                [current[begin:end], dt[begin:end, None]], axis=1
            ).astype(np.float32, copy=False)
            model_tensor = torch.from_numpy(model_input).to(args.device)
            if separate_species_input:
                species_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        current[begin:end, 2:],
                        dtype=np.float64,
                    )
                ).to(args.device)
                result = model(model_tensor, species_tensor)
            else:
                result = model(model_tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            predicted_delta_parts.append(
                result.detach().cpu().numpy().astype(np.float64, copy=False)
            )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    model_output = np.concatenate(predicted_delta_parts, axis=0)
    true_delta = target[:, 2:] - current[:, 2:]
    (
        predicted_temperature_delta,
        predicted_total_enthalpy_delta,
        predicted_delta,
    ) = _split_model_output(
        model_output,
        true_delta.shape[1],
        manifest,
    )
    predicted_target = current[:, 2:] + predicted_delta
    current_species = current[:, 2:]
    target_species = target[:, 2:]
    error = predicted_delta - true_delta
    gas = ct.Solution(str(args.artifact_dir / manifest["mechanism"]))
    mechanism_species_names = list(manifest["species_names"])
    if tuple(map(canonical_species_name, mechanism_species_names)) != tuple(
        map(canonical_species_name, species_names)
    ):
        raise ValueError("Artifact mechanism and evaluation species orders differ")
    formation_enthalpy_mass = np.asarray(
        [
            gas.species(name).thermo.h(298.15)
            / gas.molecular_weights[gas.species_index(name)]
            for name in mechanism_species_names
        ],
        dtype=np.float64,
    )
    predicted_formation_energy = -(predicted_delta @ formation_enthalpy_mass)
    true_formation_energy = -(true_delta @ formation_enthalpy_mass)
    formation_energy_error = predicted_formation_energy - true_formation_energy
    energy_active = np.abs(true_formation_energy) > 1.0e-12
    energy_relative_error = np.abs(formation_energy_error[energy_active]) / np.maximum(
        np.abs(true_formation_energy[energy_active]), 1.0e-300
    )
    true_activity = np.max(np.abs(true_delta), axis=1)
    reactive = true_activity > args.reactive_threshold

    row_pred_norm = np.linalg.norm(predicted_delta, axis=1)
    row_true_norm = np.linalg.norm(true_delta, axis=1)
    row_denominator = row_pred_norm * row_true_norm
    valid_cosine = reactive & (row_denominator > 0.0)
    row_cosine = np.full(current.shape[0], np.nan, dtype=np.float64)
    row_cosine[valid_cosine] = np.sum(
        predicted_delta[valid_cosine] * true_delta[valid_cosine], axis=1
    ) / row_denominator[valid_cosine]

    absolute_error_sum = float(np.sum(np.abs(error)))
    true_absolute_sum = float(np.sum(np.abs(true_delta)))
    normalized_mae = absolute_error_sum / max(true_absolute_sum, 1.0e-300)
    predicted_source = predicted_delta / dt[:, None]
    true_source = true_delta / dt[:, None]
    source_error = predicted_source - true_source
    source_nmae = float(
        np.sum(np.abs(source_error))
        / max(float(np.sum(np.abs(true_source))), 1.0e-300)
    )
    molecular_weights = np.asarray(
        [
            gas.molecular_weights[gas.species_index(name)]
            for name in mechanism_species_names
        ],
        dtype=np.float64,
    )
    predicted_mixture_weight = np.reciprocal(
        np.sum(predicted_target / molecular_weights[None, :], axis=1)
    )
    target_mixture_weight = np.reciprocal(
        np.sum(target_species / molecular_weights[None, :], axis=1)
    )
    mixture_weight_error = predicted_mixture_weight - target_mixture_weight
    mixture_weight_nmae = float(
        np.sum(np.abs(mixture_weight_error))
        / max(float(np.sum(np.abs(target_mixture_weight))), 1.0e-300)
    )
    global_cosine = _safe_cosine(predicted_delta[reactive], true_delta[reactive])
    sign_mask = np.abs(true_delta) > args.sign_threshold
    sign_agreement = float(
        np.mean(
            np.sign(predicted_delta[sign_mask])
            == np.sign(true_delta[sign_mask])
        )
    )
    row_true_l1 = np.sum(np.abs(true_delta), axis=1)
    row_error_l1 = np.sum(np.abs(error), axis=1)
    active_scale = np.maximum(row_true_l1, args.sign_threshold)
    row_relative_l1 = row_error_l1 / active_scale

    temperature_report = {"predicted": False}
    temperature_gate_pass = True
    if predicted_temperature_delta is not None:
        true_temperature_delta = target[:, 0] - current[:, 0]
        predicted_temperature = current[:, 0] + predicted_temperature_delta
        temperature_error = predicted_temperature_delta - true_temperature_delta
        absolute_temperature_error = np.abs(temperature_error)
        worst_temperature_order = np.argsort(
            -absolute_temperature_error
        )[: min(100, current.shape[0])]
        temperature_delta_nmae = float(
            np.sum(np.abs(temperature_error))
            / max(float(np.sum(np.abs(true_temperature_delta))), 1.0e-300)
        )
        temperature_gate_pass = bool(
            np.all(np.isfinite(predicted_temperature))
            and np.all(predicted_temperature >= args.minimum_temperature)
            and np.all(predicted_temperature <= args.maximum_temperature)
            and temperature_delta_nmae <= args.max_temperature_delta_nmae
        )
        temperature_report = {
            "predicted": True,
            "delta_mae_K": float(np.mean(np.abs(temperature_error))),
            "delta_rmse_K": float(np.sqrt(np.mean(np.square(temperature_error)))),
            "delta_true_mean_abs_K": float(
                np.mean(np.abs(true_temperature_delta))
            ),
            "delta_predicted_mean_abs_K": float(
                np.mean(np.abs(predicted_temperature_delta))
            ),
            "delta_normalized_mae": temperature_delta_nmae,
            "delta_cosine": _safe_cosine(
                predicted_temperature_delta, true_temperature_delta
            ),
            "next_temperature_mae_K": float(
                np.mean(np.abs(predicted_temperature - target[:, 0]))
            ),
            "absolute_error_p90_K": float(
                np.quantile(absolute_temperature_error, 0.90)
            ),
            "absolute_error_p99_K": float(
                np.quantile(absolute_temperature_error, 0.99)
            ),
            "absolute_error_p999_K": float(
                np.quantile(absolute_temperature_error, 0.999)
            ),
            "absolute_error_max_K": float(
                np.max(absolute_temperature_error)
            ),
            "predicted_minimum_K": float(np.min(predicted_temperature)),
            "predicted_maximum_K": float(np.max(predicted_temperature)),
            "target_minimum_K": float(np.min(target[:, 0])),
            "target_maximum_K": float(np.max(target[:, 0])),
            "worst_rows": [
                {
                    "row": int(index),
                    "source_index": int(source_index[index]),
                    "temperature_K": float(current[index, 0]),
                    "true_delta_temperature_K": float(
                        true_temperature_delta[index]
                    ),
                    "predicted_delta_temperature_K": float(
                        predicted_temperature_delta[index]
                    ),
                    "absolute_error_K": float(
                        absolute_temperature_error[index]
                    ),
                    "species_delta_l1": float(
                        np.sum(np.abs(true_delta[index]))
                    ),
                }
                for index in worst_temperature_order
            ],
        }

    total_enthalpy_report = {"predicted": False}
    total_enthalpy_gate_pass = True
    if predicted_total_enthalpy_delta is not None:
        if h_total_before is None or h_total_after is None:
            raise ValueError(
                "delta_h_total artifact requires h_total_before/h_total_after labels"
            )
        true_total_enthalpy_delta = h_total_after - h_total_before
        total_enthalpy_error = (
            predicted_total_enthalpy_delta - true_total_enthalpy_delta
        )
        total_enthalpy_nmae = float(
            np.sum(np.abs(total_enthalpy_error))
            / max(float(np.sum(np.abs(true_total_enthalpy_delta))), 1.0e-300)
        )
        total_enthalpy_gate_pass = bool(
            np.all(np.isfinite(predicted_total_enthalpy_delta))
            and total_enthalpy_nmae <= args.max_total_enthalpy_delta_nmae
        )
        total_enthalpy_report = {
            "predicted": True,
            "units": "J/kg per chemistry interval",
            "delta_mae": float(np.mean(np.abs(total_enthalpy_error))),
            "delta_rmse": float(np.sqrt(np.mean(np.square(total_enthalpy_error)))),
            "delta_true_mean_abs": float(
                np.mean(np.abs(true_total_enthalpy_delta))
            ),
            "delta_predicted_mean_abs": float(
                np.mean(np.abs(predicted_total_enthalpy_delta))
            ),
            "delta_normalized_mae": total_enthalpy_nmae,
            "delta_cosine": _safe_cosine(
                predicted_total_enthalpy_delta, true_total_enthalpy_delta
            ),
        }

    activity_edges = np.unique(
        np.quantile(true_activity, np.linspace(0.0, 1.0, 11))
    )
    activity_bins = []
    for lower, upper in zip(activity_edges[:-1], activity_edges[1:]):
        mask = (true_activity >= lower) & (true_activity <= upper)
        item = _subset_metrics(predicted_delta, true_delta, row_cosine, mask)
        item.update({"lower": float(lower), "upper": float(upper)})
        activity_bins.append(item)

    temperature_edges = np.asarray(
        [0.0, 500.0, 900.0, 1200.0, 1500.0, 1800.0, 2200.0, np.inf]
    )
    temperature_bins = []
    for lower, upper in zip(temperature_edges[:-1], temperature_edges[1:]):
        mask = (current[:, 0] >= lower) & (current[:, 0] < upper)
        item = _subset_metrics(predicted_delta, true_delta, row_cosine, mask)
        item.update({"lower_K": float(lower), "upper_K": float(upper)})
        temperature_bins.append(item)

    worst_order = np.argsort(-row_error_l1)[: min(100, current.shape[0])]
    worst_rows = [
        {
            "row": int(index),
            "source_index": int(source_index[index]),
            "temperature_K": float(current[index, 0]),
            "pressure_Pa": float(current[index, 1]),
            "true_activity": float(true_activity[index]),
            "true_delta_l1": float(row_true_l1[index]),
            "error_l1": float(row_error_l1[index]),
            "relative_l1": float(row_relative_l1[index]),
            "delta_cosine": float(row_cosine[index]),
        }
        for index in worst_order
    ]

    per_species = []
    for index, species in enumerate(species_names):
        species_true = true_delta[:, index]
        species_error = error[:, index]
        denominator = float(np.sum(np.abs(species_true)))
        per_species.append(
            {
                "species": species,
                "delta_mae": float(np.mean(np.abs(species_error))),
                "true_delta_mean_abs": float(np.mean(np.abs(species_true))),
                "normalized_absolute_error": float(
                    np.sum(np.abs(species_error)) / max(denominator, 1.0e-300)
                ),
                "cosine": _safe_cosine(
                    predicted_delta[:, index], species_true
                ),
            }
        )
    per_species.sort(
        key=lambda item: item["delta_mae"], reverse=True
    )

    element_matrix = np.asarray(
        [
            [
                gas.n_atoms(
                    gas.species_index(mechanism_species_names[species_index]),
                    element_index,
                )
                / molecular_weights[species_index]
                for species_index in range(gas.n_species)
            ]
            for element_index in range(gas.n_elements)
        ],
        dtype=np.float64,
    )
    element_residual = predicted_delta @ element_matrix.T
    mass_sum_error = np.abs(np.sum(predicted_target, axis=1) - 1.0)
    mean_abs_element_residual = float(np.mean(np.abs(element_residual)))
    max_abs_element_residual = float(np.max(np.abs(element_residual)))
    gate_pass = (
        normalized_mae <= args.max_normalized_mae
        and np.isfinite(global_cosine)
        and global_cosine >= args.minimum_global_cosine
        and not np.any(predicted_target < -args.positivity_tolerance)
        and temperature_gate_pass
        and total_enthalpy_gate_pass
        and source_nmae <= args.max_source_nmae
        and mixture_weight_nmae <= args.max_mixture_molecular_weight_nmae
        and float(np.max(mass_sum_error)) <= args.max_mass_sum_error
        and max_abs_element_residual <= args.max_element_residual
    )
    report = {
        "artifact_dir": str(args.artifact_dir),
        "source": str(args.source),
        "n_pairs": int(current.shape[0]),
        "n_species": len(species_names),
        "reactive_threshold": args.reactive_threshold,
        "reactive_pair_fraction": float(np.mean(reactive)),
        "delta_y": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "true_mean_abs": float(np.mean(np.abs(true_delta))),
            "predicted_mean_abs": float(np.mean(np.abs(predicted_delta))),
            "normalized_mae": normalized_mae,
            "global_reactive_cosine": global_cosine,
            "median_reactive_row_cosine": float(
                np.nanmedian(row_cosine[reactive])
            ),
            "p10_reactive_row_cosine": float(
                np.nanpercentile(row_cosine[reactive], 10)
            ),
            "sign_agreement": sign_agreement,
            "row_relative_l1_p50": float(np.median(row_relative_l1[reactive])),
            "row_relative_l1_p90": float(np.quantile(row_relative_l1[reactive], 0.9)),
            "row_relative_l1_p99": float(np.quantile(row_relative_l1[reactive], 0.99)),
            "row_relative_l1_max": float(np.max(row_relative_l1[reactive])),
        },
        "formation_energy": {
            "units": "J/kg per chemistry interval",
            "mae": float(np.mean(np.abs(formation_energy_error))),
            "normalized_mae": float(
                np.sum(np.abs(formation_energy_error))
                / max(float(np.sum(np.abs(true_formation_energy))), 1.0e-300)
            ),
            "cosine": _safe_cosine(
                predicted_formation_energy,
                true_formation_energy,
            ),
            "relative_error_p50": float(np.median(energy_relative_error)),
            "relative_error_p90": float(np.quantile(energy_relative_error, 0.9)),
            "relative_error_p99": float(np.quantile(energy_relative_error, 0.99)),
        },
        "species_source": {
            "units": "1/s",
            "mae": float(np.mean(np.abs(source_error))),
            "rmse": float(np.sqrt(np.mean(np.square(source_error)))),
            "true_mean_abs": float(np.mean(np.abs(true_source))),
            "normalized_mae": source_nmae,
            "cosine": _safe_cosine(predicted_source, true_source),
        },
        "mixture_molecular_weight": {
            "units": "kg/kmol",
            "mae": float(np.mean(np.abs(mixture_weight_error))),
            "normalized_mae": mixture_weight_nmae,
            "predicted_minimum": float(np.min(predicted_mixture_weight)),
            "predicted_maximum": float(np.max(predicted_mixture_weight)),
        },
        "temperature": temperature_report,
        "total_enthalpy": total_enthalpy_report,
        "activity_deciles": activity_bins,
        "temperature_bins": temperature_bins,
        "worst_rows_by_delta_l1": worst_rows,
        "composition": {
            "negative_entry_rate": float(np.mean(predicted_target < 0.0)),
            "negative_row_rate": float(
                np.mean(np.any(predicted_target < 0.0, axis=1))
            ),
            "negative_below_tolerance_entry_rate": float(
                np.mean(predicted_target < -args.positivity_tolerance)
            ),
            "negative_below_tolerance_row_rate": float(
                np.mean(
                    np.any(
                        predicted_target < -args.positivity_tolerance,
                        axis=1,
                    )
                )
            ),
            "minimum_mass_fraction": float(np.min(predicted_target)),
            "positivity_tolerance": args.positivity_tolerance,
            "input_minimum_mass_fraction": float(
                np.min(current_species)
            ),
            "input_negative_entry_rate": float(
                np.mean(current_species < 0.0)
            ),
            "target_minimum_mass_fraction": float(
                np.min(target_species)
            ),
            "target_negative_entry_rate": float(
                np.mean(target_species < 0.0)
            ),
            "mean_mass_sum_error": float(
                np.mean(mass_sum_error)
            ),
            "max_mass_sum_error": float(
                np.max(mass_sum_error)
            ),
            "mean_abs_element_residual": mean_abs_element_residual,
            "max_abs_element_residual": max_abs_element_residual,
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "microseconds_per_pair": 1.0e6 * elapsed / current.shape[0],
        },
        "gate": {
            "pass": bool(gate_pass),
            "max_normalized_mae": args.max_normalized_mae,
            "minimum_global_cosine": args.minimum_global_cosine,
            "positivity_tolerance": args.positivity_tolerance,
            "requires_nonnegative_prediction": True,
            "max_temperature_delta_nmae": args.max_temperature_delta_nmae,
            "minimum_temperature_K": args.minimum_temperature,
            "maximum_temperature_K": args.maximum_temperature,
            "temperature_pass": temperature_gate_pass,
            "total_enthalpy_pass": total_enthalpy_gate_pass,
            "max_total_enthalpy_delta_nmae": (
                args.max_total_enthalpy_delta_nmae
            ),
            "source_pass": source_nmae <= args.max_source_nmae,
            "max_source_nmae": args.max_source_nmae,
            "mixture_molecular_weight_pass": (
                mixture_weight_nmae <= args.max_mixture_molecular_weight_nmae
            ),
            "max_mixture_molecular_weight_nmae": (
                args.max_mixture_molecular_weight_nmae
            ),
            "conservation_pass": (
                float(np.max(mass_sum_error)) <= args.max_mass_sum_error
                and max_abs_element_residual <= args.max_element_residual
            ),
            "max_mass_sum_error": args.max_mass_sum_error,
            "max_element_residual": args.max_element_residual,
        },
        "per_species_by_delta_mae": per_species,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
