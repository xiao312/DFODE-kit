from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from dfode_kit.data.implicit_roots import progress_stratified_indices
from dfode_kit.data.interval_pairs import load_interval_pair_arrays
from dfode_kit.evaluation.implicit_warm_start import (
    ConstantPressureAdiabaticRHS,
    _element_matrix,
    _numeric_summary,
    _positive_segment,
    _run_cvode_reference,
    _solve_backward_euler,
)
from dfode_kit.evaluation.latent_sequence import _to_jsonable
from dfode_kit.evaluation.stoich_interval import _load_model


class StoichiometricRootPredictor:
    def __init__(self, checkpoint_path: str, device: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_type = self.checkpoint.get("model_type")
        if model_type in {
            "thermo_stoichiometric_interval",
            "latent_substep_stoichiometric_interval",
            "thermo_progress_substep_stoichiometric_interval",
        }:
            raise ValueError("implicit-root stage prediction currently requires a standard stoichiometric checkpoint")
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.state_mean = np.asarray(self.checkpoint["state_mean"], dtype=np.float64)
        self.state_std = np.asarray(self.checkpoint["state_std"], dtype=np.float64)
        self.log_dt_mean = np.asarray(self.checkpoint["log_dt_mean"], dtype=np.float64)
        self.log_dt_std = np.asarray(self.checkpoint["log_dt_std"], dtype=np.float64)
        self.state_std = np.where(self.state_std > 0.0, self.state_std, 1.0)
        self.log_dt_std = np.where(self.log_dt_std > 0.0, self.log_dt_std, 1.0)
        self.model = _load_model(self.checkpoint, self.state_mean.size, self.device)

    def predict(self, full_state: np.ndarray, dt: float) -> np.ndarray:
        normalized_state = ((full_state - self.state_mean) / self.state_std).astype(np.float32)[None, :]
        normalized_log_dt = (
            (np.log(max(float(dt), 1e-300)) - self.log_dt_mean) / self.log_dt_std
        ).astype(np.float32).reshape(1, 1)
        state_tensor = torch.tensor(normalized_state, dtype=torch.float32, device=self.device)
        dt_tensor = torch.tensor(normalized_log_dt, dtype=torch.float32, device=self.device)
        species_tensor = torch.tensor(full_state[2:][None, :], dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(state_tensor, dt_tensor, current_species=species_tensor)["next_state"]
        raw = output[0].detach().cpu().numpy()
        prediction = np.empty_like(full_state, dtype=np.float64)
        prediction[:2] = raw[:2] * self.state_std[:2] + self.state_mean[:2]
        prediction[2:] = raw[2:]
        return prediction


def _phase_name(attrs: dict, override: str | None) -> str | None:
    if override:
        return override
    value = attrs.get("phase_name")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value) if value else None


def _scale(state, rtol, species_atol, temperature_atol):
    absolute = np.full(state.size, species_atol, dtype=np.float64)
    absolute[0] = temperature_atol
    return absolute + rtol * np.abs(state)


def _wrms(values, scale):
    return float(np.sqrt(np.mean(np.square(values / scale))))


def _proposal(strategy, state, pressure, step_size, rhs, predictor, minimum_temperature, positivity_floor):
    exact_rhs = 0
    model_calls = 0
    model_seconds = 0.0
    if strategy == "hold":
        proposal = state.copy()
    elif strategy == "explicit":
        proposal = state + step_size * rhs(state)
        exact_rhs = 1
    elif strategy == "neural":
        full_state = np.concatenate(([state[0], pressure], state[1:]))
        started = time.perf_counter()
        full_prediction = predictor.predict(full_state, step_size)
        model_seconds = time.perf_counter() - started
        model_calls = 1
        proposal = np.concatenate(([full_prediction[0]], full_prediction[2:]))
    else:
        raise ValueError(f"unknown proposal strategy: {strategy}")
    proposal, alpha = _positive_segment(
        state,
        proposal,
        minimum_temperature,
        positivity_floor,
    )
    return proposal, alpha, exact_rhs, model_calls, model_seconds


def _blank_work():
    return {
        "rhs_evaluations": 0,
        "jacobian_evaluations": 0,
        "linear_solves": 0,
        "newton_iterations": 0,
        "model_calls": 0,
        "model_seconds": 0.0,
        "limiter_alphas": [],
    }


def _add_solve_work(work, solve):
    work["rhs_evaluations"] += int(solve["rhs_evaluations"])
    work["jacobian_evaluations"] += int(solve["jacobian_evaluations"])
    work["linear_solves"] += int(solve["linear_solves"])
    work["newton_iterations"] += int(solve["iterations"])


def _attempt_be(
    rhs,
    state,
    pressure,
    step_size,
    strategy,
    predictor,
    settings,
):
    work = _blank_work()
    try:
        proposal, alpha, extra_rhs, model_calls, model_seconds = _proposal(
            strategy,
            state,
            pressure,
            step_size,
            rhs,
            predictor,
            settings["minimum_temperature"],
            settings["positivity_floor"],
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None, work, "proposal_failure"
    work["rhs_evaluations"] += extra_rhs
    work["model_calls"] += model_calls
    work["model_seconds"] += model_seconds
    work["limiter_alphas"].append(alpha)
    solve = _solve_backward_euler(
        rhs,
        state,
        proposal,
        step_size,
        _scale(
            state,
            settings["relative_tolerance"],
            settings["species_absolute_tolerance"],
            settings["temperature_absolute_tolerance"],
        ),
        tolerance=settings["newton_tolerance"],
        max_iterations=settings["max_newton_iterations"],
        max_line_search_steps=settings["max_line_search_steps"],
        fd_relative_step=settings["fd_relative_step"],
        fd_species_scale=settings["fd_species_scale"],
    )
    _add_solve_work(work, solve)
    return np.asarray(solve["state"], dtype=np.float64) if solve["converged"] else None, work, solve["status"]


def _merge_work(total, local):
    for key in (
        "rhs_evaluations",
        "jacobian_evaluations",
        "linear_solves",
        "newton_iterations",
        "model_calls",
    ):
        total[key] += local[key]
    total["model_seconds"] += local["model_seconds"]
    total["limiter_alphas"].extend(local["limiter_alphas"])


def _adaptive_be(rhs, initial, pressure, total_dt, strategy, predictor, settings):
    started = time.perf_counter()
    state = initial.copy()
    remaining = float(total_dt)
    step_size = remaining
    accepted = 0
    rejected = 0
    attempts = 0
    easy_streak = 0
    work = _blank_work()
    status = "complete"
    min_accepted = None

    while remaining > max(1e-18, 1e-14 * total_dt):
        if accepted >= settings["max_accepted_steps"]:
            status = "max_accepted_steps"
            break
        if attempts >= settings["max_attempts"]:
            status = "max_attempts"
            break
        step_size = min(step_size, remaining)
        if step_size < settings["min_substep"] and remaining > settings["min_substep"]:
            status = "minimum_substep"
            break
        attempts += 1
        next_state, local_work, local_status = _attempt_be(
            rhs,
            state,
            pressure,
            step_size,
            strategy,
            predictor,
            settings,
        )
        _merge_work(work, local_work)
        if next_state is None:
            rejected += 1
            easy_streak = 0
            step_size *= 0.5
            status = local_status
            continue
        state = next_state
        remaining = max(0.0, remaining - step_size)
        accepted += 1
        min_accepted = step_size if min_accepted is None else min(min_accepted, step_size)
        if local_work["newton_iterations"] <= 2:
            easy_streak += 1
        else:
            easy_streak = 0
        if easy_streak >= 2:
            step_size = min(2.0 * step_size, remaining) if remaining > 0.0 else step_size
            easy_streak = 0
        else:
            step_size = min(step_size, remaining) if remaining > 0.0 else step_size
        status = "complete"

    return {
        "completed": status == "complete" and remaining <= max(1e-18, 1e-14 * total_dt),
        "status": status,
        "state": state,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "covered_fraction": float((total_dt - remaining) / total_dt),
        "min_accepted_step": min_accepted,
        "wall_seconds": time.perf_counter() - started,
        **{key: value for key, value in work.items() if key != "limiter_alphas"},
        "mean_limiter_alpha": float(np.mean(work["limiter_alphas"])) if work["limiter_alphas"] else None,
    }


def _attempt_sdirk2(rhs, state, pressure, step_size, strategy, predictor, settings):
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    work = _blank_work()
    stage1, stage1_work, stage1_status = _attempt_be(
        rhs,
        state,
        pressure,
        gamma * step_size,
        strategy,
        predictor,
        settings,
    )
    _merge_work(work, stage1_work)
    if stage1 is None:
        return None, None, work, f"stage1_{stage1_status}", 0, 0
    stage1_iterations = stage1_work["newton_iterations"]
    try:
        stage1_rhs = rhs(stage1)
    except (ValueError, RuntimeError, FloatingPointError):
        return None, None, work, "stage1_rhs_failure", stage1_iterations, 0
    work["rhs_evaluations"] += 1
    stage2_base = state + step_size * (1.0 - gamma) * stage1_rhs
    if not rhs.is_valid(stage2_base):
        return None, None, work, "stage2_base_invalid", stage1_iterations, 0
    stage2, stage2_work, stage2_status = _attempt_be(
        rhs,
        stage2_base,
        pressure,
        gamma * step_size,
        strategy,
        predictor,
        settings,
    )
    _merge_work(work, stage2_work)
    if stage2 is None:
        return None, None, work, f"stage2_{stage2_status}", stage1_iterations, stage2_work["newton_iterations"]
    stage2_iterations = stage2_work["newton_iterations"]
    try:
        stage2_rhs = rhs(stage2)
    except (ValueError, RuntimeError, FloatingPointError):
        return None, None, work, "stage2_rhs_failure", stage1_iterations, stage2_iterations
    work["rhs_evaluations"] += 1
    first_order = state + step_size * stage2_rhs
    error_scale = _scale(
        stage2,
        settings["relative_tolerance"],
        settings["species_absolute_tolerance"],
        settings["temperature_absolute_tolerance"],
    )
    error_norm = _wrms(stage2 - first_order, error_scale)
    return stage2, error_norm, work, "solved", stage1_iterations, stage2_iterations


def _adaptive_sdirk2(rhs, initial, pressure, total_dt, strategy, predictor, settings):
    started = time.perf_counter()
    state = initial.copy()
    remaining = float(total_dt)
    step_size = remaining
    accepted = 0
    rejected = 0
    nonlinear_rejections = 0
    error_rejections = 0
    attempts = 0
    work = _blank_work()
    status = "complete"
    min_accepted = None
    stage1_iterations = 0
    stage2_iterations = 0
    accepted_error_norms = []

    while remaining > max(1e-18, 1e-14 * total_dt):
        if accepted >= settings["max_accepted_steps"]:
            status = "max_accepted_steps"
            break
        if attempts >= settings["max_attempts"]:
            status = "max_attempts"
            break
        step_size = min(step_size, remaining)
        if step_size < settings["min_substep"] and remaining > settings["min_substep"]:
            status = "minimum_substep"
            break
        attempts += 1
        next_state, error_norm, local_work, local_status, stage1_local, stage2_local = _attempt_sdirk2(
            rhs,
            state,
            pressure,
            step_size,
            strategy,
            predictor,
            settings,
        )
        _merge_work(work, local_work)
        stage1_iterations += stage1_local
        stage2_iterations += stage2_local
        if next_state is None:
            rejected += 1
            nonlinear_rejections += 1
            step_size *= 0.5
            status = local_status
            continue
        if not np.isfinite(error_norm) or error_norm > settings["sdirk_error_tolerance"]:
            rejected += 1
            error_rejections += 1
            factor = 0.1 if not np.isfinite(error_norm) else np.clip(
                0.9 * (settings["sdirk_error_tolerance"] / max(error_norm, 1e-300)) ** 0.5,
                0.1,
                0.8,
            )
            step_size *= float(factor)
            status = "embedded_error_rejection"
            continue

        state = next_state
        remaining = max(0.0, remaining - step_size)
        accepted += 1
        min_accepted = step_size if min_accepted is None else min(min_accepted, step_size)
        accepted_error_norms.append(error_norm)
        factor = np.clip(
            0.9 * (settings["sdirk_error_tolerance"] / max(error_norm, 1e-300)) ** 0.5,
            0.5,
            2.0,
        )
        step_size = min(float(factor) * step_size, remaining) if remaining > 0.0 else step_size
        status = "complete"

    return {
        "completed": status == "complete" and remaining <= max(1e-18, 1e-14 * total_dt),
        "status": status,
        "state": state,
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "nonlinear_rejections": nonlinear_rejections,
        "error_rejections": error_rejections,
        "covered_fraction": float((total_dt - remaining) / total_dt),
        "min_accepted_step": min_accepted,
        "mean_accepted_error_norm": float(np.mean(accepted_error_norms)) if accepted_error_norms else None,
        "stage1_newton_iterations": stage1_iterations,
        "stage2_newton_iterations": stage2_iterations,
        "wall_seconds": time.perf_counter() - started,
        **{key: value for key, value in work.items() if key != "limiter_alphas"},
        "mean_limiter_alpha": float(np.mean(work["limiter_alphas"])) if work["limiter_alphas"] else None,
    }


def _finish_row(result, current, target, element_matrix):
    state = result.pop("state")
    current_y = current[1:]
    target_y = target[1:]
    result.update(
        {
            "temperature_abs_error": float(abs(state[0] - target[0])),
            "species_mae": float(np.mean(np.abs(state[1:] - target_y))),
            "negative_species_fraction": float(np.mean(state[1:] < 0.0)),
            "mass_sum_delta": float(abs(np.sum(state[1:]) - np.sum(current_y))),
            "mean_abs_element_delta": float(np.mean(np.abs(element_matrix @ (state[1:] - current_y)))),
        }
    )
    return result


def _summarize_method(rows):
    complete = [row for row in rows if row["completed"]]
    return {
        "n_samples": len(rows),
        "completion_rate": float(len(complete) / len(rows)) if rows else 0.0,
        "species_mae_completed": _numeric_summary(row["species_mae"] for row in complete),
        "temperature_abs_error_completed": _numeric_summary(row["temperature_abs_error"] for row in complete),
        "accepted_steps": _numeric_summary(row["accepted_steps"] for row in rows),
        "rejected_steps": _numeric_summary(row["rejected_steps"] for row in rows),
        "newton_iterations": _numeric_summary(row["newton_iterations"] for row in rows),
        "rhs_evaluations": _numeric_summary(row["rhs_evaluations"] for row in rows),
        "jacobian_evaluations": _numeric_summary(row["jacobian_evaluations"] for row in rows),
        "model_calls": _numeric_summary(row["model_calls"] for row in rows),
        "wall_seconds": _numeric_summary(row["wall_seconds"] for row in rows),
        "mass_sum_delta": _numeric_summary(row["mass_sum_delta"] for row in complete),
        "mean_abs_element_delta": _numeric_summary(row["mean_abs_element_delta"] for row in complete),
        "status_counts": {
            status: int(sum(row["status"] == status for row in rows))
            for status in sorted({row["status"] for row in rows})
        },
    }


def evaluate_hybrid_integrators(
    checkpoint_path: str,
    source_path: str,
    mech_path: str,
    output_path: str,
    *,
    phase_name: str | None = None,
    device: str | None = None,
    max_samples: int = 12,
    sample_seed: int = 20260713,
    max_accepted_steps: int = 256,
    min_substep: float = 1e-12,
    relative_tolerance: float = 1e-6,
    species_absolute_tolerance: float = 1e-12,
    temperature_absolute_tolerance: float = 1e-6,
    newton_tolerance: float = 1.0,
    max_newton_iterations: int = 12,
    max_line_search_steps: int = 12,
    fd_relative_step: float = 1e-5,
    fd_species_scale: float = 1e-5,
    minimum_temperature: float = 200.0,
    positivity_floor: float = 0.0,
    sdirk_error_tolerance: float = 1.0,
) -> dict:
    import cantera as ct

    current, target, dt, dt_bin, dt_bin_edges, species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
    progress = np.max(np.abs(target[:, 2:] - current[:, 2:]), axis=1)
    selected = progress_stratified_indices(dt_bin, progress, max_samples, sample_seed)
    phase_name = _phase_name(attrs, phase_name)
    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    if list(gas.species_names) != list(species_names):
        raise ValueError("mechanism species order does not match the evaluation dataset")
    element_matrix = _element_matrix(gas)
    predictor = StoichiometricRootPredictor(checkpoint_path, device=device)
    settings = {
        "max_accepted_steps": int(max_accepted_steps),
        "max_attempts": int(4 * max_accepted_steps + 128),
        "min_substep": float(min_substep),
        "relative_tolerance": float(relative_tolerance),
        "species_absolute_tolerance": float(species_absolute_tolerance),
        "temperature_absolute_tolerance": float(temperature_absolute_tolerance),
        "newton_tolerance": float(newton_tolerance),
        "max_newton_iterations": int(max_newton_iterations),
        "max_line_search_steps": int(max_line_search_steps),
        "fd_relative_step": float(fd_relative_step),
        "fd_species_scale": float(fd_species_scale),
        "minimum_temperature": float(minimum_temperature),
        "positivity_floor": float(positivity_floor),
        "sdirk_error_tolerance": float(sdirk_error_tolerance),
    }
    methods = [
        ("be_hold", "be", "hold"),
        ("be_explicit", "be", "explicit"),
        ("be_neural", "be", "neural"),
        ("sdirk2_hold", "sdirk2", "hold"),
        ("sdirk2_explicit", "sdirk2", "explicit"),
        ("sdirk2_neural", "sdirk2", "neural"),
    ]
    rows = []
    for source_index in selected:
        full_current = current[source_index]
        full_target = target[source_index]
        pressure = float(full_current[1])
        initial = np.concatenate(([full_current[0]], full_current[2:]))
        target_unknown = np.concatenate(([full_target[0]], full_target[2:]))
        rhs = ConstantPressureAdiabaticRHS(gas, pressure, minimum_temperature, positivity_floor)
        for method_name, family, strategy in methods:
            if family == "be":
                method_result = _adaptive_be(
                    rhs,
                    initial,
                    pressure,
                    float(dt[source_index]),
                    strategy,
                    predictor if strategy == "neural" else None,
                    settings,
                )
            else:
                method_result = _adaptive_sdirk2(
                    rhs,
                    initial,
                    pressure,
                    float(dt[source_index]),
                    strategy,
                    predictor if strategy == "neural" else None,
                    settings,
                )
            method_result = _finish_row(method_result, initial, target_unknown, element_matrix)
            rows.append(
                {
                    "sample_index": int(source_index),
                    "dt": float(dt[source_index]),
                    "dt_bin": int(dt_bin[source_index]),
                    "source_progress": float(progress[source_index]),
                    "method": method_name,
                    **method_result,
                }
            )

    method_metrics = {
        method_name: _summarize_method([row for row in rows if row["method"] == method_name])
        for method_name, _family, _strategy in methods
    }
    dt_bin_metrics = []
    for bin_index in sorted({row["dt_bin"] for row in rows}):
        local_rows = [row for row in rows if row["dt_bin"] == bin_index]
        dt_bin_metrics.append(
            {
                "dt_bin": int(bin_index),
                "dt_min": float(dt_bin_edges[bin_index]),
                "dt_max": float(dt_bin_edges[bin_index + 1]),
                "method_metrics": {
                    method_name: _summarize_method([row for row in local_rows if row["method"] == method_name])
                    for method_name, _family, _strategy in methods
                },
            }
        )

    current_selected = current[selected]
    target_selected = target[selected]
    dt_selected = dt[selected]
    bins_selected = dt_bin[selected]
    cvode = _run_cvode_reference(
        ct,
        mech_path,
        phase_name,
        current_selected,
        target_selected,
        dt_selected,
        selected,
        bins_selected,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=species_absolute_tolerance,
    )
    cvode_mean = cvode["advance_seconds"]["mean"]
    runtime_comparison = {"cvode_advance_seconds_mean": cvode_mean}
    for method_name, metrics in method_metrics.items():
        method_mean = metrics["wall_seconds"]["mean"]
        runtime_comparison[method_name] = {
            "wall_seconds_mean": method_mean,
            "raw_cvode_speedup": float(cvode_mean / method_mean) if method_mean else None,
            "completion_rate": metrics["completion_rate"],
            "warning": "Speedup is meaningful only when completion and accuracy match CVODE.",
        }

    result = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "mechanism": mech_path,
        "phase_name": phase_name,
        "cantera_version": ct.__version__,
        "device": str(predictor.device),
        "n_source_pairs": int(current.shape[0]),
        "n_selected_pairs": int(selected.size),
        "selected_indices": selected.tolist(),
        "species_names": list(species_names),
        "settings": settings,
        "method_metrics": method_metrics,
        "dt_bin_metrics": dt_bin_metrics,
        "runtime_comparison": runtime_comparison,
        "cvode_reference": cvode,
        "samples": rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    csv_path = output.with_name(f"{output.stem}.samples.csv")
    if rows:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    result["sample_csv"] = str(csv_path)
    return result
