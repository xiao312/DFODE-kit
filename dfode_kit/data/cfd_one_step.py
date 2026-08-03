from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from dfode_kit.data.cfd_conditioned import (
    integrate_constant_pressure_state,
    mechanism_permutation,
)


def generate_cfd_one_step_pairs(
    snapshot_path: str | Path,
    mechanism_path: str,
    output_path: str | Path,
    *,
    dt: float,
    phase_name: str | None = None,
    split: str = "train",
    energy: str = "on",
) -> dict:
    import cantera as ct

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    snapshot_path = Path(snapshot_path)
    with h5py.File(snapshot_path, "r") as source:
        cells = source["cells"]
        raw_states = np.asarray(cells["states"], dtype=np.float64)
        source_species = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in source["species_names"][:]
        ]
        provenance = {
            name: np.asarray(cells[name])
            for name in (
                "source_cell_index",
                "parent_source_cell_index",
                "interpolation_fraction",
                "is_original",
                "coordinates",
            )
            if name in cells
        }

    gas = (
        ct.Solution(mechanism_path, phase_name)
        if phase_name
        else ct.Solution(mechanism_path)
    )
    permutation = mechanism_permutation(
        source_species, gas.species_names
    )
    mass_fractions = raw_states[:, 2:][:, permutation].copy()
    if float(mass_fractions.min()) < -1e-12:
        raise ValueError("snapshot has materially negative mass fractions")
    mass_fractions[mass_fractions < 0.0] = 0.0
    mass_fractions /= mass_fractions.sum(axis=1, keepdims=True)
    current = np.column_stack([raw_states[:, :2], mass_fractions])
    midpoint = np.empty_like(current)
    target = np.empty_like(current)
    times = np.asarray([0.0, 0.5 * dt, dt], dtype=np.float64)
    failures: list[int] = []
    for index, state in enumerate(current):
        try:
            sequence = integrate_constant_pressure_state(
                gas, state, times, energy=energy
            )
            midpoint[index] = sequence[1]
            target[index] = sequence[2]
        except Exception:
            failures.append(index)

    if failures:
        keep = np.ones(current.shape[0], dtype=bool)
        keep[np.asarray(failures, dtype=np.int64)] = False
        current = current[keep]
        midpoint = midpoint[keep]
        target = target[keep]
        provenance = {
            name: values[keep] for name, values in provenance.items()
        }

    delta = target[:, 2:] - current[:, 2:]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as output:
        output.attrs["schema_version"] = "interval-pairs-v1"
        output.attrs["source_kind"] = "cfd-temperature-interpolation"
        output.attrs["source_snapshot"] = str(snapshot_path)
        output.attrs["mechanism"] = mechanism_path
        output.attrs["mechanism_id"] = mechanism_path
        output.attrs["phase_name"] = phase_name or ""
        output.attrs["cantera_version"] = ct.__version__
        output.attrs["reactor"] = "IdealGasConstPressureReactor"
        output.attrs["energy"] = energy
        output.attrs["split"] = split
        output.attrs["n_pairs"] = current.shape[0]
        output.attrs["min_dt"] = dt
        output.attrs["max_dt"] = dt
        output.attrs["n_dt_bins"] = 1
        output.attrs["failed_trajectory_count"] = len(failures)
        output.create_dataset(
            "species_names",
            data=np.asarray(gas.species_names, dtype=object),
            dtype=string_dtype,
        )
        output.create_dataset(
            "dt_bin_edges",
            data=np.asarray([0.5 * dt, 1.5 * dt], dtype=np.float64),
        )
        output.create_dataset(
            "dt_bin_counts",
            data=np.asarray([current.shape[0]], dtype=np.int64),
        )
        pairs = output.create_group("pairs")
        pairs.create_dataset(
            "current_states", data=current, compression="gzip"
        )
        pairs.create_dataset(
            "midpoint_states", data=midpoint, compression="gzip"
        )
        pairs.create_dataset(
            "target_states", data=target, compression="gzip"
        )
        pairs.create_dataset(
            "dt",
            data=np.full(current.shape[0], dt, dtype=np.float64),
            compression="gzip",
        )
        pairs.create_dataset(
            "midpoint_dt",
            data=np.full(current.shape[0], 0.5 * dt, dtype=np.float64),
            compression="gzip",
        )
        pairs.create_dataset(
            "dt_bin",
            data=np.zeros(current.shape[0], dtype=np.int32),
            compression="gzip",
        )
        pairs.create_dataset(
            "max_abs_delta_y",
            data=np.max(np.abs(delta), axis=1),
            compression="gzip",
        )
        pairs.create_dataset(
            "temperature_delta",
            data=target[:, 0] - current[:, 0],
            compression="gzip",
        )
        for name, values in provenance.items():
            pairs.create_dataset(name, data=values, compression="gzip")
    return {
        "snapshot": str(snapshot_path),
        "output": str(output_path),
        "split": split,
        "dt": dt,
        "input_states": int(raw_states.shape[0]),
        "written_pairs": int(current.shape[0]),
        "failures": len(failures),
        "max_abs_delta_y": float(np.max(np.abs(delta))),
        "reactive_pair_fraction_1e-8": float(
            np.mean(np.max(np.abs(delta), axis=1) >= 1e-8)
        ),
        "temperature_delta_abs_max_k": float(
            np.max(np.abs(target[:, 0] - current[:, 0]))
        ),
    }
