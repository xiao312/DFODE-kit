from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_STEP_TEMPERATURE_DELTA = 1.0
DEFAULT_STEP_SPECIES_DELTA = 1e-6
DEFAULT_TRAJECTORY_TEMPERATURE_RISE = 50.0
DEFAULT_REACTIVE_STEP_FRACTION = 0.05


def reactive_step_mask(
    states: np.ndarray,
    *,
    step_temperature_delta: float = DEFAULT_STEP_TEMPERATURE_DELTA,
    step_species_delta: float = DEFAULT_STEP_SPECIES_DELTA,
) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"Expected states shaped (n_times, 2 + n_species), got {states.shape}")
    if states.shape[0] < 2:
        return np.zeros((0,), dtype=bool)

    deltas = np.diff(states, axis=0)
    temperature_steps = np.abs(deltas[:, 0]) >= step_temperature_delta
    species_steps = np.max(np.abs(deltas[:, 2:]), axis=1) >= step_species_delta
    return temperature_steps | species_steps


def summarize_trajectory_reactivity(
    states: np.ndarray,
    times: np.ndarray,
    *,
    step_temperature_delta: float = DEFAULT_STEP_TEMPERATURE_DELTA,
    step_species_delta: float = DEFAULT_STEP_SPECIES_DELTA,
    trajectory_temperature_rise: float = DEFAULT_TRAJECTORY_TEMPERATURE_RISE,
    reactive_step_fraction_threshold: float = DEFAULT_REACTIVE_STEP_FRACTION,
) -> dict[str, float | bool]:
    states = np.asarray(states, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if states.shape[0] != times.shape[0]:
        raise ValueError(f"states/time length mismatch: {states.shape[0]} vs {times.shape[0]}")

    mask = reactive_step_mask(
        states,
        step_temperature_delta=step_temperature_delta,
        step_species_delta=step_species_delta,
    )
    dts = np.diff(times)
    deltas = np.diff(states, axis=0)
    temperature_rise = float(np.max(states[:, 0]) - states[0, 0])
    max_abs_step_temperature_delta = float(np.max(np.abs(deltas[:, 0]))) if deltas.size else 0.0
    max_abs_step_species_delta = float(np.max(np.abs(deltas[:, 2:]))) if deltas.size else 0.0
    max_abs_temperature_rate = float(np.max(np.abs(deltas[:, 0]) / np.maximum(dts, 1e-300))) if dts.size else 0.0
    max_abs_species_rate = float(np.max(np.max(np.abs(deltas[:, 2:]), axis=1) / np.maximum(dts, 1e-300))) if dts.size else 0.0
    reactive_step_fraction = float(np.mean(mask)) if mask.size else 0.0
    reactive_state_fraction = float(np.count_nonzero(np.r_[False, mask] | np.r_[mask, False]) / states.shape[0])
    is_reactive = bool(
        temperature_rise >= trajectory_temperature_rise
        or reactive_step_fraction >= reactive_step_fraction_threshold
    )

    return {
        "temperature_initial": float(states[0, 0]),
        "temperature_final": float(states[-1, 0]),
        "temperature_max": float(np.max(states[:, 0])),
        "temperature_rise": temperature_rise,
        "max_abs_step_temperature_delta": max_abs_step_temperature_delta,
        "max_abs_step_species_delta": max_abs_step_species_delta,
        "max_abs_temperature_rate": max_abs_temperature_rate,
        "max_abs_species_rate": max_abs_species_rate,
        "reactive_step_fraction": reactive_step_fraction,
        "reactive_state_fraction": reactive_state_fraction,
        "is_reactive": is_reactive,
    }


def summarize_sequence_file_reactivity(
    input_path: str,
    *,
    step_temperature_delta: float = DEFAULT_STEP_TEMPERATURE_DELTA,
    step_species_delta: float = DEFAULT_STEP_SPECIES_DELTA,
    trajectory_temperature_rise: float = DEFAULT_TRAJECTORY_TEMPERATURE_RISE,
    reactive_step_fraction_threshold: float = DEFAULT_REACTIVE_STEP_FRACTION,
) -> dict:
    trajectory_summaries = {}
    with h5py.File(input_path, "r") as h5:
        for name, group in h5["trajectories"].items():
            summary = summarize_trajectory_reactivity(
                group["states"][:],
                group["times"][:],
                step_temperature_delta=step_temperature_delta,
                step_species_delta=step_species_delta,
                trajectory_temperature_rise=trajectory_temperature_rise,
                reactive_step_fraction_threshold=reactive_step_fraction_threshold,
            )
            summary["attrs"] = {
                key: value.item() if hasattr(value, "item") else value
                for key, value in group.attrs.items()
            }
            trajectory_summaries[name] = summary

        reactive_flags = np.array([item["is_reactive"] for item in trajectory_summaries.values()], dtype=bool)
        step_fractions = np.array([item["reactive_step_fraction"] for item in trajectory_summaries.values()], dtype=np.float64)
        state_fractions = np.array([item["reactive_state_fraction"] for item in trajectory_summaries.values()], dtype=np.float64)
        temperature_rises = np.array([item["temperature_rise"] for item in trajectory_summaries.values()], dtype=np.float64)
        attrs = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in h5.attrs.items()
        }

    aggregate = {
        "input_path": input_path,
        "attrs": attrs,
        "thresholds": {
            "step_temperature_delta": step_temperature_delta,
            "step_species_delta": step_species_delta,
            "trajectory_temperature_rise": trajectory_temperature_rise,
            "reactive_step_fraction_threshold": reactive_step_fraction_threshold,
        },
        "n_trajectories": int(len(trajectory_summaries)),
        "n_reactive_trajectories": int(np.count_nonzero(reactive_flags)),
        "reactive_trajectory_fraction": float(np.mean(reactive_flags)) if reactive_flags.size else 0.0,
        "mean_reactive_step_fraction": float(np.mean(step_fractions)) if step_fractions.size else 0.0,
        "mean_reactive_state_fraction": float(np.mean(state_fractions)) if state_fractions.size else 0.0,
        "median_temperature_rise": float(np.median(temperature_rises)) if temperature_rises.size else 0.0,
        "max_temperature_rise": float(np.max(temperature_rises)) if temperature_rises.size else 0.0,
    }
    return {
        "aggregate": aggregate,
        "trajectories": trajectory_summaries,
    }


def write_reactivity_summary(input_path: str, output_path: str, **kwargs) -> dict:
    summary = summarize_sequence_file_reactivity(input_path, **kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary
