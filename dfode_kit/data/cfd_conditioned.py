from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

CFD_SNAPSHOT_SCHEMA_VERSION = "cfd-cell-snapshot-v1"
CFD_SEQUENCE_SCHEMA_VERSION = "sequence-v2"


def canonical_species_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def _decode_names(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def load_cfd_snapshot(path: str | Path) -> dict:
    with h5py.File(path, "r") as handle:
        schema = str(handle.attrs.get("schema_version", ""))
        if schema != CFD_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"expected {CFD_SNAPSHOT_SCHEMA_VERSION}, got {schema!r}"
            )
        cells = handle["cells"]
        result = {
            "states": np.asarray(cells["states"], dtype=np.float64),
            "coordinates": np.asarray(
                cells["coordinates"], dtype=np.float64
            ),
            "source_cell_index": np.asarray(
                cells["source_cell_index"], dtype=np.int64
            ),
            "stratum": np.asarray(cells["stratum"], dtype=np.int32),
            "sampling_probability": np.asarray(
                cells["sampling_probability"], dtype=np.float64
            ),
            "species_names": _decode_names(handle["species_names"][:]),
            "attrs": dict(handle.attrs),
        }
        for name in ("density", "velocity", "reaction_rate"):
            if name in cells:
                result[name] = np.asarray(cells[name], dtype=np.float64)
    states = result["states"]
    if states.ndim != 2 or states.shape[1] != 2 + len(
        result["species_names"]
    ):
        raise ValueError("snapshot state width does not match species names")
    if not np.all(np.isfinite(states)):
        raise ValueError("snapshot contains non-finite states")
    if np.any(states[:, 0] <= 0.0) or np.any(states[:, 1] <= 0.0):
        raise ValueError("snapshot temperature and pressure must be positive")
    return result


def mechanism_permutation(
    source_species: Sequence[str],
    mechanism_species: Sequence[str],
) -> np.ndarray:
    source = {
        canonical_species_name(name): index
        for index, name in enumerate(source_species)
    }
    if len(source) != len(source_species):
        raise ValueError("source species names are ambiguous after canonicalization")
    missing = [
        name
        for name in mechanism_species
        if canonical_species_name(name) not in source
    ]
    if missing:
        raise ValueError(f"snapshot is missing mechanism species: {missing}")
    extras = sorted(
        set(source)
        - {canonical_species_name(name) for name in mechanism_species}
    )
    if extras:
        raise ValueError(f"snapshot contains species absent from mechanism: {extras}")
    return np.asarray(
        [source[canonical_species_name(name)] for name in mechanism_species],
        dtype=np.int64,
    )


def prepare_initial_states(
    snapshot: dict,
    mechanism_species: Sequence[str],
    *,
    negative_tolerance: float = 1e-12,
) -> np.ndarray:
    permutation = mechanism_permutation(
        snapshot["species_names"], mechanism_species
    )
    raw = np.asarray(snapshot["states"], dtype=np.float64)
    mass_fractions = raw[:, 2:][:, permutation].copy()
    if float(np.min(mass_fractions)) < -negative_tolerance:
        raise ValueError(
            "snapshot contains mass fractions below the negative tolerance"
        )
    mass_fractions[mass_fractions < 0.0] = 0.0
    sums = mass_fractions.sum(axis=1)
    if np.any(sums <= 0.0):
        raise ValueError("snapshot contains a state with zero total mass fraction")
    mass_fractions /= sums[:, None]
    return np.column_stack([raw[:, :2], mass_fractions])


def stratified_split(
    strata: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    validation: list[np.ndarray] = []
    singletons: list[int] = []
    for label in np.unique(strata):
        indexes = np.flatnonzero(strata == label)
        if indexes.shape[0] == 1:
            singletons.append(int(indexes[0]))
            continue
        indexes = indexes[rng.permutation(indexes.shape[0])]
        count = int(round(validation_fraction * indexes.shape[0]))
        count = min(max(count, 1), indexes.shape[0] - 1)
        validation.append(indexes[:count])
        train.append(indexes[count:])
    target_validation = min(
        max(int(round(validation_fraction * strata.shape[0])), 1),
        strata.shape[0] - 1,
    )
    current_validation = sum(item.shape[0] for item in validation)
    shuffled_singletons = np.asarray(singletons, dtype=np.int64)
    if shuffled_singletons.size:
        shuffled_singletons = shuffled_singletons[
            rng.permutation(shuffled_singletons.shape[0])
        ]
        singleton_validation_count = min(
            max(target_validation - current_validation, 0),
            shuffled_singletons.shape[0],
        )
        validation.append(
            shuffled_singletons[:singleton_validation_count]
        )
        train.append(
            shuffled_singletons[singleton_validation_count:]
        )
    train_indexes = np.sort(np.concatenate(train))
    validation_nonempty = [item for item in validation if item.size]
    if not validation_nonempty:
        raise ValueError("snapshot is too small to construct a validation split")
    validation_indexes = np.sort(np.concatenate(validation_nonempty))
    return train_indexes, validation_indexes


def cfd_time_grid(
    *,
    min_time: float,
    max_time: float,
    steps: int,
) -> np.ndarray:
    if min_time <= 0.0 or max_time <= min_time:
        raise ValueError("expected 0 < min_time < max_time")
    if steps < 2:
        raise ValueError("steps must be at least two")
    values = np.empty(steps + 1, dtype=np.float64)
    values[0] = 0.0
    values[1:] = np.geomspace(min_time, max_time, steps)
    return values


def integrate_constant_pressure_state(
    gas,
    initial_state: np.ndarray,
    times: np.ndarray,
    *,
    energy: str,
) -> np.ndarray:
    import cantera as ct

    gas.TPY = (
        float(initial_state[0]),
        float(initial_state[1]),
        initial_state[2:],
    )
    reactor = ct.IdealGasConstPressureReactor(
        gas, energy=energy, clone=False
    )
    network = ct.ReactorNet([reactor])
    sequence = np.empty(
        (times.shape[0], initial_state.shape[0]), dtype=np.float64
    )
    sequence[0] = np.asarray([gas.T, gas.P, *gas.Y], dtype=np.float64)
    for index, time_value in enumerate(times[1:], start=1):
        network.advance(float(time_value))
        sequence[index] = np.asarray(
            [gas.T, gas.P, *gas.Y], dtype=np.float64
        )
    return sequence


def _write_split(
    output_path: str | Path,
    *,
    snapshot_path: str | Path,
    snapshot: dict,
    initial_states: np.ndarray,
    indexes: np.ndarray,
    mech_path: str,
    phase_name: str | None,
    mechanism_id: str,
    times: np.ndarray,
    split: str,
    energy: str,
) -> dict:
    import cantera as ct

    gas = (
        ct.Solution(mech_path, phase_name)
        if phase_name
        else ct.Solution(mech_path)
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    written = 0
    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = CFD_SEQUENCE_SCHEMA_VERSION
        handle.attrs["mechanism"] = mech_path
        handle.attrs["mechanism_id"] = mechanism_id
        handle.attrs["phase_name"] = phase_name or ""
        handle.attrs["cantera_version"] = ct.__version__
        handle.attrs["reactor"] = "IdealGasConstPressureReactor"
        handle.attrs["energy"] = energy
        handle.attrs["sampling"] = "log-time"
        handle.attrs["split"] = split
        handle.attrs["source_kind"] = "cfd-conditioned"
        handle.attrs["source_snapshot"] = str(snapshot_path)
        handle.attrs["steps"] = int(times.shape[0] - 1)
        handle.create_dataset(
            "species_names",
            data=np.asarray(gas.species_names, dtype=object),
            dtype=string_dtype,
        )
        mechanisms = handle.create_group("mechanisms")
        mechanism = mechanisms.create_group("000000")
        mechanism.attrs["mechanism_id"] = mechanism_id
        mechanism.attrs["mechanism"] = mech_path
        mechanism.attrs["phase_name"] = phase_name or ""
        mechanism.create_dataset(
            "species_names",
            data=np.asarray(gas.species_names, dtype=object),
            dtype=string_dtype,
        )
        sequences = handle.create_group("sequences")
        trajectories = handle.create_group("trajectories")
        for local_index, snapshot_index in enumerate(indexes):
            try:
                sequence = integrate_constant_pressure_state(
                    gas,
                    initial_states[snapshot_index],
                    times,
                    energy=energy,
                )
            except Exception as error:
                failures.append(
                    {
                        "snapshot_index": int(snapshot_index),
                        "source_cell_index": int(
                            snapshot["source_cell_index"][snapshot_index]
                        ),
                        "error": str(error),
                    }
                )
                continue
            name = f"{written:06d}"
            sequences.create_dataset(
                name, data=sequence, compression="gzip"
            )
            trajectory = trajectories.create_group(name)
            trajectory.create_dataset(
                "states", data=sequence, compression="gzip"
            )
            trajectory.create_dataset("times", data=times)
            for target in (sequences[name], trajectory):
                target.attrs["mechanism_id"] = mechanism_id
                target.attrs["mechanism"] = mech_path
                target.attrs["phase_name"] = phase_name or ""
                target.attrs["sampling"] = "log-time"
                target.attrs["split"] = split
                target.attrs["source_kind"] = "cfd-conditioned"
                target.attrs["snapshot_index"] = int(snapshot_index)
                target.attrs["source_cell_index"] = int(
                    snapshot["source_cell_index"][snapshot_index]
                )
                target.attrs["stratum"] = int(
                    snapshot["stratum"][snapshot_index]
                )
                target.attrs["sampling_probability"] = float(
                    snapshot["sampling_probability"][snapshot_index]
                )
                target.attrs["coordinates"] = snapshot[
                    "coordinates"
                ][snapshot_index]
            written += 1
        handle.attrs["trajectory_count"] = written
        handle.attrs["failed_trajectory_count"] = len(failures)
        handle.create_dataset(
            "source_snapshot_index", data=indexes.astype(np.int64)
        )
    return {
        "output": str(output_path),
        "split": split,
        "requested_trajectories": int(indexes.shape[0]),
        "written_trajectories": written,
        "failed_trajectories": failures,
    }


def generate_cfd_conditioned_sequences(
    snapshot_path: str | Path,
    mech_path: str,
    train_output: str | Path,
    validation_output: str | Path,
    *,
    phase_name: str | None = None,
    mechanism_id: str | None = None,
    validation_fraction: float = 0.2,
    seed: int = 20260728,
    min_time: float = 1e-9,
    max_time: float = 1e-3,
    steps: int = 96,
    energy: str = "on",
) -> dict:
    import cantera as ct

    snapshot = load_cfd_snapshot(snapshot_path)
    gas = (
        ct.Solution(mech_path, phase_name)
        if phase_name
        else ct.Solution(mech_path)
    )
    initial_states = prepare_initial_states(
        snapshot, gas.species_names
    )
    train_indexes, validation_indexes = stratified_split(
        snapshot["stratum"],
        validation_fraction=validation_fraction,
        seed=seed,
    )
    times = cfd_time_grid(
        min_time=min_time, max_time=max_time, steps=steps
    )
    mech_id = mechanism_id or mech_path
    train = _write_split(
        train_output,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        initial_states=initial_states,
        indexes=train_indexes,
        mech_path=mech_path,
        phase_name=phase_name,
        mechanism_id=mech_id,
        times=times,
        split="train",
        energy=energy,
    )
    validation = _write_split(
        validation_output,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        initial_states=initial_states,
        indexes=validation_indexes,
        mech_path=mech_path,
        phase_name=phase_name,
        mechanism_id=mech_id,
        times=times,
        split="val",
        energy=energy,
    )
    return {
        "schema_version": CFD_SEQUENCE_SCHEMA_VERSION,
        "snapshot": str(snapshot_path),
        "mechanism": mech_path,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "times": times.tolist(),
        "train": train,
        "validation": validation,
    }


def merge_interval_pair_datasets(
    sources: Sequence[str | Path],
    output_path: str | Path,
    *,
    labels: Sequence[str] | None = None,
    split: str | None = None,
) -> dict:
    if len(sources) < 1:
        raise ValueError("at least one source dataset is required")
    source_paths = [Path(path) for path in sources]
    source_labels = (
        list(labels)
        if labels is not None
        else [path.stem for path in source_paths]
    )
    if len(source_labels) != len(source_paths):
        raise ValueError("labels must match the number of sources")

    handles = [h5py.File(path, "r") for path in source_paths]
    try:
        species = handles[0]["species_names"][:]
        edges = np.asarray(handles[0]["dt_bin_edges"], dtype=np.float64)
        for path, handle in zip(source_paths[1:], handles[1:]):
            if not np.array_equal(species, handle["species_names"][:]):
                raise ValueError(f"species order differs in {path}")
            if not np.allclose(
                edges,
                np.asarray(handle["dt_bin_edges"], dtype=np.float64),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(f"dt bin edges differ in {path}")

        common = set(handles[0]["pairs"].keys())
        for handle in handles[1:]:
            common &= set(handle["pairs"].keys())
        pair_counts = [
            int(handle["pairs/current_states"].shape[0])
            for handle in handles
        ]
        merge_names = []
        for name in sorted(common):
            datasets = [handle[f"pairs/{name}"] for handle in handles]
            if all(
                dataset.ndim >= 1
                and dataset.shape[0] == count
                and dataset.shape[1:] == datasets[0].shape[1:]
                for dataset, count in zip(datasets, pair_counts)
            ):
                merge_names.append(name)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(output_path, "w") as output:
            for key, value in handles[0].attrs.items():
                output.attrs[key] = value
            output.attrs["schema_version"] = "interval-pairs-v1"
            output.attrs["source_kind"] = "merged"
            output.attrs["source_paths"] = json.dumps(
                [str(path) for path in source_paths]
            )
            output.attrs["source_labels"] = json.dumps(source_labels)
            output.attrs["n_pairs"] = int(sum(pair_counts))
            if split is not None:
                output.attrs["split"] = split
            output.create_dataset("species_names", data=species)
            output.create_dataset("dt_bin_edges", data=edges)
            output.create_dataset(
                "dt_bin_counts",
                data=np.sum(
                    [
                        np.asarray(handle["dt_bin_counts"], dtype=np.int64)
                        for handle in handles
                    ],
                    axis=0,
                ),
            )
            output.create_dataset(
                "source_dataset_labels",
                data=np.asarray(source_labels, dtype=object),
                dtype=string_dtype,
            )
            pairs = output.create_group("pairs")
            for name in merge_names:
                pairs.create_dataset(
                    name,
                    data=np.concatenate(
                        [
                            np.asarray(handle[f"pairs/{name}"])
                            for handle in handles
                        ],
                        axis=0,
                    ),
                    compression="gzip",
                )
            pairs.create_dataset(
                "source_dataset_index",
                data=np.concatenate(
                    [
                        np.full(count, index, dtype=np.int16)
                        for index, count in enumerate(pair_counts)
                    ]
                ),
                compression="gzip",
            )
    finally:
        for handle in handles:
            handle.close()

    return {
        "output": str(output_path),
        "n_pairs": int(sum(pair_counts)),
        "pair_counts": dict(zip(source_labels, pair_counts)),
        "merged_pair_fields": merge_names,
    }
