from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from dfode_kit.data.interval_pairs import load_interval_pair_arrays
from dfode_kit.evaluation.implicit_warm_start import (
    ConstantPressureAdiabaticRHS,
    _positive_segment,
    _solve_backward_euler,
)
from dfode_kit.evaluation.latent_sequence import _to_jsonable


def progress_stratified_indices(
    dt_bin: np.ndarray,
    progress: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Select approximately equal counts per dt bin and across progress quantiles."""
    n_samples = int(dt_bin.size)
    if max_samples <= 0 or max_samples >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    active_bins = np.asarray(sorted(np.unique(dt_bin).tolist()), dtype=np.int64)
    if max_samples < active_bins.size:
        positions = np.linspace(0, active_bins.size - 1, max_samples)
        active_bins = active_bins[np.unique(np.rint(positions).astype(np.int64))]

    base = max_samples // active_bins.size
    remainder = max_samples % active_bins.size
    selected = []
    for bin_offset, bin_index in enumerate(active_bins):
        quota = base + int(bin_offset < remainder)
        candidates = np.flatnonzero(dt_bin == bin_index)
        rng.shuffle(candidates)
        candidates = candidates[np.argsort(progress[candidates], kind="stable")]
        quota = min(quota, candidates.size)
        if quota == 1:
            local = np.asarray([candidates.size // 2], dtype=np.int64)
        else:
            local = np.unique(np.rint(np.linspace(0, candidates.size - 1, quota)).astype(np.int64))
        selected.extend(candidates[local].tolist())

    if len(selected) < max_samples:
        selected_set = set(selected)
        remaining = np.asarray([i for i in range(n_samples) if i not in selected_set], dtype=np.int64)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)].tolist())
    return np.asarray(sorted(selected[:max_samples]), dtype=np.int64)


def _phase_name(attrs: dict, override: str | None) -> str | None:
    if override:
        return override
    value = attrs.get("phase_name")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value) if value else None


def _newton_scale(
    state: np.ndarray,
    relative_tolerance: float,
    species_absolute_tolerance: float,
    temperature_absolute_tolerance: float,
) -> np.ndarray:
    absolute = np.full(state.size, species_absolute_tolerance, dtype=np.float64)
    absolute[0] = temperature_absolute_tolerance
    return absolute + relative_tolerance * np.abs(state)


def _write_root_pairs(
    output_path: str,
    current_states: np.ndarray,
    target_states: np.ndarray,
    step_sizes: np.ndarray,
    source_indices: np.ndarray,
    substep_indices: np.ndarray,
    species_names: list[str],
    reaction_reversible: np.ndarray,
    *,
    source_path: str,
    mechanism: str,
    phase_name: str | None,
    cantera_version: str,
    split: str,
    n_dt_bins: int,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if step_sizes.size == 0:
        raise RuntimeError("adaptive continuation produced no accepted implicit roots")

    dt_min = float(np.min(step_sizes))
    dt_max = float(np.max(step_sizes))
    if dt_min == dt_max:
        dt_min *= 0.99
        dt_max *= 1.01
    edges = np.geomspace(max(dt_min * (1.0 - 1e-12), 1e-300), dt_max * (1.0 + 1e-12), n_dt_bins + 1)
    bins = np.searchsorted(edges, step_sizes, side="right") - 1
    bins = np.clip(bins, 0, n_dt_bins - 1).astype(np.int32)
    counts = np.bincount(bins, minlength=n_dt_bins).astype(np.int64)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output, "w") as handle:
        handle.attrs["schema_version"] = "implicit-root-pairs-v1"
        handle.attrs["source_interval_path"] = str(source_path)
        handle.attrs["mechanism"] = str(mechanism)
        handle.attrs["phase_name"] = phase_name or ""
        handle.attrs["reactor"] = "adiabatic-constant-pressure"
        handle.attrs["implicit_method"] = "backward-euler"
        handle.attrs["cantera_version"] = cantera_version
        handle.attrs["split"] = split
        handle.attrs["n_pairs"] = int(step_sizes.size)
        handle.attrs["n_dt_bins"] = int(n_dt_bins)
        handle.attrs["min_dt"] = float(np.min(step_sizes))
        handle.attrs["max_dt"] = float(np.max(step_sizes))
        handle.attrs["generation_settings"] = json.dumps(_to_jsonable(settings), sort_keys=True)
        handle.create_dataset("species_names", data=np.asarray(species_names, dtype=object), dtype=string_dtype)
        handle.create_dataset("reaction_reversible", data=reaction_reversible.astype(np.uint8))
        handle.create_dataset("dt_bin_edges", data=edges.astype(np.float64))
        handle.create_dataset("dt_bin_counts", data=counts)
        pairs = handle.create_group("pairs")
        pairs.create_dataset("current_states", data=current_states.astype(np.float64), compression="gzip")
        pairs.create_dataset("target_states", data=target_states.astype(np.float64), compression="gzip")
        pairs.create_dataset("dt", data=step_sizes.astype(np.float64), compression="gzip")
        pairs.create_dataset("dt_bin", data=bins, compression="gzip")
        pairs.create_dataset("source_trajectory_index", data=source_indices.astype(np.int32), compression="gzip")
        pairs.create_dataset("source_substep_index", data=substep_indices.astype(np.int32), compression="gzip")
        pairs.create_dataset(
            "max_abs_delta_y",
            data=np.max(np.abs(target_states[:, 2:] - current_states[:, 2:]), axis=1).astype(np.float64),
            compression="gzip",
        )
        pairs.create_dataset(
            "temperature_delta",
            data=(target_states[:, 0] - current_states[:, 0]).astype(np.float64),
            compression="gzip",
        )
    return edges, counts


def generate_implicit_root_dataset(
    source_path: str,
    output_path: str,
    mech_path: str,
    *,
    phase_name: str | None = None,
    max_source_samples: int = 256,
    sample_seed: int = 20260713,
    max_accepted_steps: int = 128,
    min_substep: float = 1e-12,
    n_dt_bins: int = 12,
    relative_tolerance: float = 1e-6,
    species_absolute_tolerance: float = 1e-12,
    temperature_absolute_tolerance: float = 1e-6,
    newton_tolerance: float = 1e-4,
    max_newton_iterations: int = 16,
    max_line_search_steps: int = 16,
    fd_relative_step: float = 1e-5,
    fd_species_scale: float = 1e-5,
    minimum_temperature: float = 200.0,
    positivity_floor: float = 0.0,
) -> dict:
    import cantera as ct

    current, target, dt, dt_bin, _edges, species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
    progress = np.max(np.abs(target[:, 2:] - current[:, 2:]), axis=1)
    selected = progress_stratified_indices(dt_bin, progress, max_source_samples, sample_seed)
    phase_name = _phase_name(attrs, phase_name)
    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    if list(gas.species_names) != list(species_names):
        raise ValueError("mechanism species order does not match the source interval dataset")

    root_current = []
    root_target = []
    root_dt = []
    root_source = []
    root_substep = []
    interval_rows = []
    total_rhs_evaluations = 0
    total_jacobian_evaluations = 0
    total_rejected_steps = 0

    for source_index in selected:
        source_state = current[source_index]
        reference_target = target[source_index]
        pressure = float(source_state[1])
        state = np.concatenate(([source_state[0]], source_state[2:])).astype(np.float64)
        rhs = ConstantPressureAdiabaticRHS(gas, pressure, minimum_temperature, positivity_floor)
        total_dt = float(dt[source_index])
        remaining = total_dt
        step_size = total_dt
        accepted = 0
        rejected = 0
        attempts = 0
        easy_streak = 0
        status = "complete"

        while remaining > max(1e-18, 1e-14 * total_dt):
            if accepted >= max_accepted_steps:
                status = "max_accepted_steps"
                break
            if attempts >= 4 * max_accepted_steps + 64:
                status = "max_attempts"
                break
            step_size = min(step_size, remaining)
            if step_size < min_substep and remaining > min_substep:
                status = "minimum_substep"
                break
            attempts += 1

            try:
                explicit_guess = state + step_size * rhs(state)
                total_rhs_evaluations += 1
            except (ValueError, RuntimeError, FloatingPointError):
                explicit_guess = state.copy()
            explicit_guess, _alpha = _positive_segment(
                state,
                explicit_guess,
                minimum_temperature,
                positivity_floor,
            )
            scale = _newton_scale(
                state,
                relative_tolerance,
                species_absolute_tolerance,
                temperature_absolute_tolerance,
            )
            solve = _solve_backward_euler(
                rhs,
                state,
                explicit_guess,
                step_size,
                scale,
                tolerance=newton_tolerance,
                max_iterations=max_newton_iterations,
                max_line_search_steps=max_line_search_steps,
                fd_relative_step=fd_relative_step,
                fd_species_scale=fd_species_scale,
            )
            total_rhs_evaluations += int(solve["rhs_evaluations"])
            total_jacobian_evaluations += int(solve["jacobian_evaluations"])
            if not solve["converged"]:
                rejected += 1
                total_rejected_steps += 1
                easy_streak = 0
                step_size *= 0.5
                continue

            next_state = np.asarray(solve["state"], dtype=np.float64)
            full_current = np.concatenate(([state[0], pressure], state[1:]))
            full_target = np.concatenate(([next_state[0], pressure], next_state[1:]))
            root_current.append(full_current)
            root_target.append(full_target)
            root_dt.append(step_size)
            root_source.append(int(source_index))
            root_substep.append(accepted)
            state = next_state
            remaining = max(0.0, remaining - step_size)
            accepted += 1
            if solve["iterations"] <= 2 and rejected == 0:
                easy_streak += 1
            else:
                easy_streak = 0
            if easy_streak >= 2:
                step_size = min(2.0 * step_size, remaining) if remaining > 0.0 else step_size
                easy_streak = 0
            else:
                step_size = min(step_size, remaining) if remaining > 0.0 else step_size

        final_full = np.concatenate(([state[0], pressure], state[1:]))
        interval_rows.append(
            {
                "source_index": int(source_index),
                "dt": total_dt,
                "source_progress": float(progress[source_index]),
                "accepted_steps": int(accepted),
                "rejected_steps": int(rejected),
                "completed": status == "complete",
                "status": status,
                "covered_fraction": float((total_dt - remaining) / total_dt),
                "temperature_abs_error_vs_cvode": float(abs(final_full[0] - reference_target[0])),
                "species_mae_vs_cvode": float(np.mean(np.abs(final_full[2:] - reference_target[2:]))),
            }
        )

    root_current_array = np.asarray(root_current, dtype=np.float64)
    root_target_array = np.asarray(root_target, dtype=np.float64)
    root_dt_array = np.asarray(root_dt, dtype=np.float64)
    settings = {
        "max_source_samples": max_source_samples,
        "sample_seed": sample_seed,
        "max_accepted_steps": max_accepted_steps,
        "min_substep": min_substep,
        "relative_tolerance": relative_tolerance,
        "species_absolute_tolerance": species_absolute_tolerance,
        "temperature_absolute_tolerance": temperature_absolute_tolerance,
        "newton_tolerance": newton_tolerance,
        "max_newton_iterations": max_newton_iterations,
        "max_line_search_steps": max_line_search_steps,
        "fd_relative_step": fd_relative_step,
        "fd_species_scale": fd_species_scale,
    }
    split = str(attrs.get("split", "unknown"))
    reaction_reversible = np.asarray([reaction.reversible for reaction in gas.reactions()], dtype=bool)
    root_edges, root_counts = _write_root_pairs(
        output_path,
        root_current_array,
        root_target_array,
        root_dt_array,
        np.asarray(root_source, dtype=np.int64),
        np.asarray(root_substep, dtype=np.int64),
        list(species_names),
        reaction_reversible,
        source_path=source_path,
        mechanism=mech_path,
        phase_name=phase_name,
        cantera_version=ct.__version__,
        split=split,
        n_dt_bins=n_dt_bins,
        settings=settings,
    )

    completed_rows = [row for row in interval_rows if row["completed"]]
    result = {
        "source": source_path,
        "output": output_path,
        "mechanism": mech_path,
        "phase_name": phase_name,
        "selected_source_indices": selected.tolist(),
        "n_selected_intervals": int(selected.size),
        "n_completed_intervals": len(completed_rows),
        "completion_rate": float(len(completed_rows) / selected.size) if selected.size else 0.0,
        "n_root_pairs": int(root_dt_array.size),
        "root_dt_min": float(np.min(root_dt_array)),
        "root_dt_max": float(np.max(root_dt_array)),
        "root_dt_bin_edges": root_edges.tolist(),
        "root_dt_bin_counts": root_counts.tolist(),
        "total_rhs_evaluations": int(total_rhs_evaluations),
        "total_jacobian_evaluations": int(total_jacobian_evaluations),
        "total_rejected_steps": int(total_rejected_steps),
        "completed_species_mae_vs_cvode_mean": (
            float(np.mean([row["species_mae_vs_cvode"] for row in completed_rows])) if completed_rows else None
        ),
        "completed_temperature_abs_error_vs_cvode_mean": (
            float(np.mean([row["temperature_abs_error_vs_cvode"] for row in completed_rows])) if completed_rows else None
        ),
        "settings": settings,
        "intervals": interval_rows,
    }
    summary_path = Path(output_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    result["summary_path"] = str(summary_path)
    return result
