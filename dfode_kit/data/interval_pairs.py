from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from dfode_kit.physics.atom_conservation import reaction_stoichiometry

INTERVAL_PAIR_SCHEMA_VERSION = "interval-pairs-v1"


@dataclass(frozen=True)
class IntervalPairConfig:
    min_dt: float = 1e-9
    max_dt: float = 1e-3
    n_dt_bins: int = 12
    max_pairs_per_trajectory_per_bin: int = 256
    seed: int = 20260626
    normalize_mass_fractions: bool = True


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _normalize_state_mass_fractions(states: np.ndarray) -> np.ndarray:
    normalized = np.array(states, dtype=np.float64, copy=True)
    y = normalized[:, 2:]
    sums = y.sum(axis=1, keepdims=True)
    valid = np.abs(sums) > 0.0
    y[valid[:, 0]] = y[valid[:, 0]] / sums[valid[:, 0]]
    normalized[:, 2:] = y
    return normalized


def write_interval_pair_dataset(
    source_path: str,
    output_path: str,
    *,
    config: IntervalPairConfig | None = None,
    split: str | None = None,
) -> dict:
    cfg = config or IntervalPairConfig()
    if cfg.min_dt <= 0 or cfg.max_dt <= cfg.min_dt:
        raise ValueError("Expected 0 < min_dt < max_dt")
    if cfg.n_dt_bins < 1:
        raise ValueError("n_dt_bins must be positive")

    rng = np.random.default_rng(cfg.seed)
    bin_edges = np.geomspace(cfg.min_dt, cfg.max_dt, cfg.n_dt_bins + 1)

    current_chunks = []
    target_chunks = []
    dt_chunks = []
    bin_chunks = []
    temperature_delta_chunks = []
    max_abs_delta_y_chunks = []
    source_index_chunks = []
    source_start_chunks = []
    source_mid_chunks = []
    source_end_chunks = []
    midpoint_chunks = []
    midpoint_dt_chunks = []
    pair_count_by_bin = np.zeros(cfg.n_dt_bins, dtype=np.int64)

    with h5py.File(source_path, "r") as source:
        species_names = source["species_names"][:]
        if "trajectories" not in source:
            raise ValueError("Interval-pair conversion requires sequence-v2 trajectories")

        for trajectory_index, (_name, trajectory) in enumerate(sorted(source["trajectories"].items())):
            states = np.asarray(trajectory["states"][:], dtype=np.float64)
            times = np.asarray(trajectory["times"][:], dtype=np.float64)
            if cfg.normalize_mass_fractions:
                states = _normalize_state_mass_fractions(states)

            rows_current = []
            rows_target = []
            rows_dt = []
            rows_bin = []
            rows_temperature_delta = []
            rows_max_abs_delta_y = []
            rows_source_index = []
            rows_source_start = []
            rows_source_mid = []
            rows_source_end = []
            rows_midpoint = []
            rows_midpoint_dt = []

            for bin_index in range(cfg.n_dt_bins):
                lo = bin_edges[bin_index]
                hi = bin_edges[bin_index + 1]
                candidates = []
                for start in range(times.shape[0] - 1):
                    deltas = times[start + 1 :] - times[start]
                    if bin_index == cfg.n_dt_bins - 1:
                        local_ends = np.where((deltas >= lo) & (deltas <= hi))[0] + start + 1
                    else:
                        local_ends = np.where((deltas >= lo) & (deltas < hi))[0] + start + 1
                    candidates.extend((start, int(end)) for end in local_ends)

                if not candidates:
                    continue
                if cfg.max_pairs_per_trajectory_per_bin > 0 and len(candidates) > cfg.max_pairs_per_trajectory_per_bin:
                    chosen = rng.choice(len(candidates), size=cfg.max_pairs_per_trajectory_per_bin, replace=False)
                    candidates = [candidates[int(idx)] for idx in chosen]

                for start, end in candidates:
                    current = states[start]
                    target = states[end]
                    midpoint_time = times[start] + 0.5 * (times[end] - times[start])
                    local_mid = int(np.argmin(np.abs(times[start : end + 1] - midpoint_time)) + start)
                    if local_mid <= start and end > start:
                        local_mid = start + 1
                    if local_mid >= end and end > start + 1:
                        local_mid = end - 1
                    midpoint = states[local_mid]
                    rows_current.append(current)
                    rows_target.append(target)
                    rows_dt.append(times[end] - times[start])
                    rows_bin.append(bin_index)
                    rows_temperature_delta.append(target[0] - current[0])
                    rows_max_abs_delta_y.append(np.max(np.abs(target[2:] - current[2:])))
                    rows_source_index.append(trajectory_index)
                    rows_source_start.append(start)
                    rows_source_mid.append(local_mid)
                    rows_source_end.append(end)
                    rows_midpoint.append(midpoint)
                    rows_midpoint_dt.append(times[local_mid] - times[start])
                    pair_count_by_bin[bin_index] += 1

            if rows_current:
                current_chunks.append(np.asarray(rows_current, dtype=np.float64))
                target_chunks.append(np.asarray(rows_target, dtype=np.float64))
                dt_chunks.append(np.asarray(rows_dt, dtype=np.float64))
                bin_chunks.append(np.asarray(rows_bin, dtype=np.int32))
                temperature_delta_chunks.append(np.asarray(rows_temperature_delta, dtype=np.float64))
                max_abs_delta_y_chunks.append(np.asarray(rows_max_abs_delta_y, dtype=np.float64))
                source_index_chunks.append(np.asarray(rows_source_index, dtype=np.int32))
                source_start_chunks.append(np.asarray(rows_source_start, dtype=np.int32))
                source_mid_chunks.append(np.asarray(rows_source_mid, dtype=np.int32))
                source_end_chunks.append(np.asarray(rows_source_end, dtype=np.int32))
                midpoint_chunks.append(np.asarray(rows_midpoint, dtype=np.float64))
                midpoint_dt_chunks.append(np.asarray(rows_midpoint_dt, dtype=np.float64))

        if not current_chunks:
            raise ValueError(f"No interval pairs generated from {source_path}")

        current_states = np.concatenate(current_chunks, axis=0)
        target_states = np.concatenate(target_chunks, axis=0)
        dt = np.concatenate(dt_chunks, axis=0)
        dt_bin = np.concatenate(bin_chunks, axis=0)
        temperature_delta = np.concatenate(temperature_delta_chunks, axis=0)
        max_abs_delta_y = np.concatenate(max_abs_delta_y_chunks, axis=0)
        source_trajectory_index = np.concatenate(source_index_chunks, axis=0)
        source_start_index = np.concatenate(source_start_chunks, axis=0)
        source_midpoint_index = np.concatenate(source_mid_chunks, axis=0)
        source_end_index = np.concatenate(source_end_chunks, axis=0)
        midpoint_states = np.concatenate(midpoint_chunks, axis=0)
        midpoint_dt = np.concatenate(midpoint_dt_chunks, axis=0)

        with h5py.File(output_path, "w") as target:
            _copy_attrs(source, target)
            target.attrs["schema_version"] = INTERVAL_PAIR_SCHEMA_VERSION
            target.attrs["source_sequence_path"] = source_path
            target.attrs["split"] = split or _decode(source.attrs.get("split", ""))
            target.attrs["min_dt"] = cfg.min_dt
            target.attrs["max_dt"] = cfg.max_dt
            target.attrs["n_dt_bins"] = cfg.n_dt_bins
            target.attrs["max_pairs_per_trajectory_per_bin"] = cfg.max_pairs_per_trajectory_per_bin
            target.attrs["seed"] = cfg.seed
            target.attrs["normalize_mass_fractions"] = int(cfg.normalize_mass_fractions)
            target.attrs["n_pairs"] = int(current_states.shape[0])
            target.create_dataset("species_names", data=species_names)
            target.create_dataset("dt_bin_edges", data=bin_edges)
            target.create_dataset("dt_bin_counts", data=pair_count_by_bin)
            pairs = target.create_group("pairs")
            pairs.create_dataset("current_states", data=current_states)
            pairs.create_dataset("target_states", data=target_states)
            pairs.create_dataset("dt", data=dt)
            pairs.create_dataset("dt_bin", data=dt_bin)
            pairs.create_dataset("temperature_delta", data=temperature_delta)
            pairs.create_dataset("max_abs_delta_y", data=max_abs_delta_y)
            pairs.create_dataset("source_trajectory_index", data=source_trajectory_index)
            pairs.create_dataset("source_start_index", data=source_start_index)
            pairs.create_dataset("source_midpoint_index", data=source_midpoint_index)
            pairs.create_dataset("source_end_index", data=source_end_index)
            pairs.create_dataset("midpoint_states", data=midpoint_states)
            pairs.create_dataset("midpoint_dt", data=midpoint_dt)

    return {
        "n_pairs": int(current_states.shape[0]),
        "dt_min": float(np.min(dt)),
        "dt_max": float(np.max(dt)),
        "dt_bin_counts": pair_count_by_bin.tolist(),
    }


def load_interval_pair_arrays(path: str, *, dtype=np.float64):
    with h5py.File(path, "r") as h5:
        pairs = h5["pairs"]
        current = pairs["current_states"][:].astype(dtype)
        target = pairs["target_states"][:].astype(dtype)
        dt = pairs["dt"][:].astype(np.float64)
        dt_bin = pairs["dt_bin"][:].astype(np.int32)
        dt_bin_edges = h5["dt_bin_edges"][:].astype(np.float64)
        species_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5["species_names"][:]
        ]
        attrs = dict(h5.attrs)
    return current, target, dt, dt_bin, dt_bin_edges, species_names, attrs


def load_interval_midpoint_arrays(path: str, *, dtype=np.float64):
    with h5py.File(path, "r") as h5:
        pairs = h5["pairs"]
        if "midpoint_states" not in pairs or "midpoint_dt" not in pairs:
            return None, None
        midpoint = pairs["midpoint_states"][:].astype(dtype)
        midpoint_dt = pairs["midpoint_dt"][:].astype(np.float64)
    return midpoint, midpoint_dt


def load_interval_thermo_arrays(path: str, *, dtype=np.float32):
    with h5py.File(path, "r") as h5:
        pairs = h5["pairs"]
        if "affinity_hat" not in pairs or "log_reactant_activity" not in pairs:
            raise ValueError(
                f"{path} does not contain interval thermo features. "
                "Run generate-interval-thermo-features first."
            )
        affinity_hat = pairs["affinity_hat"][:].astype(dtype)
        log_reactant_activity = pairs["log_reactant_activity"][:].astype(dtype)
        reversible = h5["reaction_reversible"][:].astype(bool)
    return affinity_hat, log_reactant_activity, reversible


def add_interval_thermo_features(
    path: str,
    mech_path: str,
    *,
    phase_name: str | None = None,
    chunk_size: int = 4096,
    mole_fraction_floor: float = 1e-300,
) -> dict:
    import cantera as ct

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    with h5py.File(path, "r+") as h5:
        pairs = h5["pairs"]
        current = pairs["current_states"]
        attrs_phase = _decode(h5.attrs.get("phase_name", ""))
        selected_phase = phase_name or attrs_phase or None
        gas = ct.Solution(mech_path, selected_phase) if selected_phase else ct.Solution(mech_path)
        stoich = reaction_stoichiometry(gas)

        n_pairs = current.shape[0]
        n_reactions = gas.n_reactions
        if "affinity_hat" in pairs:
            del pairs["affinity_hat"]
        if "log_reactant_activity" in pairs:
            del pairs["log_reactant_activity"]
        if "reaction_reversible" in h5:
            del h5["reaction_reversible"]
        affinity_ds = pairs.create_dataset("affinity_hat", shape=(n_pairs, n_reactions), dtype="f4")
        log_activity_ds = pairs.create_dataset("log_reactant_activity", shape=(n_pairs, n_reactions), dtype="f4")
        h5.create_dataset("reaction_reversible", data=stoich.reversible.astype(np.uint8))
        h5.attrs["thermo_feature_mechanism"] = mech_path
        h5.attrs["thermo_feature_phase_name"] = selected_phase or ""
        h5.attrs["thermo_feature_mole_fraction_floor"] = mole_fraction_floor

        for start in range(0, n_pairs, chunk_size):
            stop = min(start + chunk_size, n_pairs)
            states = np.asarray(current[start:stop], dtype=np.float64)
            affinity = np.empty((stop - start, n_reactions), dtype=np.float32)
            log_activity = np.empty((stop - start, n_reactions), dtype=np.float32)
            for row_idx, state in enumerate(states):
                gas.TPY = float(state[0]), float(state[1]), state[2:]
                mu_hat = np.asarray(gas.chemical_potentials, dtype=np.float64) / (ct.gas_constant * gas.T)
                x = np.clip(np.asarray(gas.X, dtype=np.float64), mole_fraction_floor, None)
                affinity[row_idx] = (-(mu_hat @ stoich.net)).astype(np.float32)
                log_activity[row_idx] = (np.log(x) @ stoich.reactants).astype(np.float32)
            affinity_ds[start:stop] = affinity
            log_activity_ds[start:stop] = log_activity

    return {
        "n_pairs": int(n_pairs),
        "n_reactions": int(n_reactions),
        "phase_name": selected_phase or "",
    }
