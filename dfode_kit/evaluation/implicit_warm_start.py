from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from dfode_kit.evaluation.latent_sequence import _to_jsonable
from dfode_kit.evaluation.stoich_interval import evaluate_stoich_interval_model


def _text_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    return text if text else None


def _stratified_indices(dt_bin: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    n_samples = int(dt_bin.shape[0])
    if max_samples <= 0 or max_samples >= n_samples:
        return np.arange(n_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    pools = []
    for value in sorted(np.unique(dt_bin).tolist()):
        indices = np.flatnonzero(dt_bin == value)
        rng.shuffle(indices)
        pools.append(indices.tolist())

    selected = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for pool in pools:
            if offset < len(pool):
                selected.append(pool[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _numeric_summary(values) -> dict:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "max": float(np.max(finite)),
    }


class ConstantPressureAdiabaticRHS:
    """Cantera RHS for x = [T, Y_1, ..., Y_K] at fixed pressure."""

    def __init__(self, gas, pressure: float, minimum_temperature: float, positivity_floor: float):
        self.gas = gas
        self.pressure = float(pressure)
        self.minimum_temperature = float(minimum_temperature)
        self.positivity_floor = float(positivity_floor)

    def is_valid(self, state: np.ndarray) -> bool:
        state = np.asarray(state, dtype=np.float64)
        return bool(
            state.ndim == 1
            and state.shape[0] == self.gas.n_species + 1
            and np.all(np.isfinite(state))
            and state[0] >= self.minimum_temperature
            and np.all(state[1:] >= self.positivity_floor)
            and np.sum(state[1:]) > 0.0
        )

    def __call__(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if not self.is_valid(state):
            raise ValueError("invalid thermochemical state")

        temperature = float(state[0])
        mass_fractions = state[1:]
        self.gas.TP = temperature, self.pressure
        self.gas.set_unnormalized_mass_fractions(mass_fractions)

        production_rates = np.asarray(self.gas.net_production_rates, dtype=np.float64)
        density = float(self.gas.density)
        species_rhs = self.gas.molecular_weights * production_rates / density
        temperature_rhs = -float(np.dot(self.gas.partial_molar_enthalpies, production_rates)) / (
            density * float(self.gas.cp_mass)
        )
        return np.concatenate(([temperature_rhs], species_rhs)).astype(np.float64, copy=False)


def _wrms(values: np.ndarray, scale: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values / scale))))


def _solve_backward_euler(
    rhs: ConstantPressureAdiabaticRHS,
    current: np.ndarray,
    guess: np.ndarray,
    dt: float,
    scale: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    max_line_search_steps: int,
    fd_relative_step: float,
    fd_species_scale: float,
) -> dict:
    started = time.perf_counter()
    rhs_evaluations = 0
    jacobian_evaluations = 0
    linear_solves = 0

    def evaluate(state):
        nonlocal rhs_evaluations
        if not rhs.is_valid(state):
            return None
        try:
            source = rhs(state)
        except (ValueError, RuntimeError, FloatingPointError):
            return None
        rhs_evaluations += 1
        residual = state - current - dt * source
        if not np.all(np.isfinite(residual)):
            return None
        return residual

    state = np.asarray(guess, dtype=np.float64).copy()
    residual = evaluate(state)
    if residual is None:
        return {
            "valid_initial": False,
            "converged": False,
            "status": "invalid_initial",
            "iterations": 0,
            "initial_wrms": None,
            "one_newton_wrms": None,
            "final_wrms": None,
            "first_jacobian_condition": None,
            "rhs_evaluations": rhs_evaluations,
            "jacobian_evaluations": jacobian_evaluations,
            "linear_solves": linear_solves,
            "wall_seconds": time.perf_counter() - started,
            "state": None,
        }

    initial_wrms = _wrms(residual, scale)
    current_wrms = initial_wrms
    first_jacobian_condition = None
    one_newton_wrms = None
    accepted_iterations = 0
    status = "max_iterations"
    converged = current_wrms <= tolerance
    if converged:
        status = "converged_initial"

    for iteration in range(max_iterations):
        if converged:
            break

        jacobian = np.empty((state.size, state.size), dtype=np.float64)
        jacobian_failed = False
        for column in range(state.size):
            floor = 1.0 if column == 0 else fd_species_scale
            step = fd_relative_step * max(abs(float(state[column])), floor)
            trial = state.copy()
            trial[column] += step
            trial_residual = evaluate(trial)
            if trial_residual is None:
                jacobian_failed = True
                break
            jacobian[:, column] = (trial_residual - residual) / step
        jacobian_evaluations += 1
        if jacobian_failed or not np.all(np.isfinite(jacobian)):
            status = "jacobian_failure"
            break

        scaled_jacobian = (jacobian * scale[None, :]) / scale[:, None]
        if first_jacobian_condition is None:
            try:
                first_jacobian_condition = _finite_or_none(np.linalg.cond(scaled_jacobian))
            except np.linalg.LinAlgError:
                first_jacobian_condition = None
        try:
            scaled_update = np.linalg.solve(scaled_jacobian, -residual / scale)
        except np.linalg.LinAlgError:
            status = "linear_solve_failure"
            break
        linear_solves += 1
        update = scale * scaled_update

        accepted = False
        damping = 1.0
        for _line_search in range(max_line_search_steps + 1):
            trial = state + damping * update
            trial_residual = evaluate(trial)
            if trial_residual is not None:
                trial_wrms = _wrms(trial_residual, scale)
                if trial_wrms < current_wrms:
                    state = trial
                    residual = trial_residual
                    current_wrms = trial_wrms
                    accepted = True
                    break
            damping *= 0.5
        if not accepted:
            status = "line_search_failure"
            break

        accepted_iterations = iteration + 1
        if accepted_iterations == 1:
            one_newton_wrms = current_wrms
        converged = current_wrms <= tolerance
        if converged:
            status = "converged"

    return {
        "valid_initial": True,
        "converged": bool(converged),
        "status": status,
        "iterations": int(accepted_iterations),
        "initial_wrms": float(initial_wrms),
        "one_newton_wrms": _finite_or_none(one_newton_wrms) if one_newton_wrms is not None else None,
        "final_wrms": float(current_wrms),
        "first_jacobian_condition": first_jacobian_condition,
        "rhs_evaluations": int(rhs_evaluations),
        "jacobian_evaluations": int(jacobian_evaluations),
        "linear_solves": int(linear_solves),
        "wall_seconds": float(time.perf_counter() - started),
        "state": state,
    }


def _positive_segment(
    current: np.ndarray,
    proposal: np.ndarray,
    minimum_temperature: float,
    positivity_floor: float,
    safety: float = 0.999,
) -> tuple[np.ndarray, float]:
    delta = proposal - current
    alpha = 1.0
    lower_bounds = np.concatenate(([minimum_temperature], np.full(current.size - 1, positivity_floor)))
    decreasing = delta < 0.0
    if np.any(decreasing):
        admissible = (current[decreasing] - lower_bounds[decreasing]) / (-delta[decreasing])
        alpha = min(alpha, safety * float(np.min(admissible)))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return current + alpha * delta, alpha


def _element_matrix(gas) -> np.ndarray:
    return np.asarray(
        [
            [gas.n_atoms(species, element) / gas.molecular_weights[species] for species in range(gas.n_species)]
            for element in range(gas.n_elements)
        ],
        dtype=np.float64,
    )


def _state_diagnostics(
    state: np.ndarray | None,
    current: np.ndarray,
    target: np.ndarray,
    element_matrix: np.ndarray,
) -> dict:
    if state is None or not np.all(np.isfinite(state)):
        return {
            "temperature_abs_error": None,
            "species_mae": None,
            "negative_species_fraction": None,
            "mass_sum_delta": None,
            "mean_abs_element_delta": None,
        }
    y = state[1:]
    current_y = current[1:]
    target_y = target[1:]
    return {
        "temperature_abs_error": float(abs(state[0] - target[0])),
        "species_mae": float(np.mean(np.abs(y - target_y))),
        "negative_species_fraction": float(np.mean(y < 0.0)),
        "mass_sum_delta": float(abs(np.sum(y) - np.sum(current_y))),
        "mean_abs_element_delta": float(np.mean(np.abs(element_matrix @ (y - current_y)))),
    }


def _summarize_guess_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"n_samples": 0}
    converged = [row for row in rows if row["converged"]]
    valid = [row for row in rows if row["valid_initial"]]
    return {
        "n_samples": len(rows),
        "initial_valid_rate": float(len(valid) / len(rows)),
        "convergence_rate": float(len(converged) / len(rows)),
        "initial_wrms": _numeric_summary(row["initial_wrms"] for row in rows),
        "one_newton_wrms": _numeric_summary(row["one_newton_wrms"] for row in rows),
        "one_newton_reduction_ratio": _numeric_summary(row["one_newton_reduction_ratio"] for row in rows),
        "final_wrms": _numeric_summary(row["final_wrms"] for row in rows),
        "iterations_when_converged": _numeric_summary(row["iterations"] for row in converged),
        "initial_species_mae": _numeric_summary(row["initial_species_mae"] for row in rows),
        "final_species_mae": _numeric_summary(row["final_species_mae"] for row in rows),
        "initial_temperature_abs_error": _numeric_summary(row["initial_temperature_abs_error"] for row in rows),
        "final_temperature_abs_error": _numeric_summary(row["final_temperature_abs_error"] for row in rows),
        "final_mass_sum_delta": _numeric_summary(row["final_mass_sum_delta"] for row in rows),
        "final_mean_abs_element_delta": _numeric_summary(row["final_mean_abs_element_delta"] for row in rows),
        "newton_wall_seconds": _numeric_summary(row["newton_wall_seconds"] for row in rows),
        "rhs_evaluations": _numeric_summary(row["rhs_evaluations"] for row in rows),
        "first_jacobian_condition": _numeric_summary(row["first_jacobian_condition"] for row in rows),
        "status_counts": {
            status: int(sum(row["status"] == status for row in rows))
            for status in sorted({row["status"] for row in rows})
        },
    }


def _guess_comparison(rows: list[dict], left: str, right: str) -> dict:
    left_rows = {row["sample_index"]: row for row in rows if row["guess"] == left}
    right_rows = {row["sample_index"]: row for row in rows if row["guess"] == right}
    common = sorted(set(left_rows) & set(right_rows))
    residual_ratios = []
    iteration_deltas = []
    left_better = 0
    both_finite = 0
    for index in common:
        lhs = left_rows[index]
        rhs = right_rows[index]
        if lhs["initial_wrms"] is not None and rhs["initial_wrms"] is not None and rhs["initial_wrms"] > 0.0:
            residual_ratios.append(lhs["initial_wrms"] / rhs["initial_wrms"])
            left_better += int(lhs["initial_wrms"] < rhs["initial_wrms"])
            both_finite += 1
        if lhs["converged"] and rhs["converged"]:
            iteration_deltas.append(lhs["iterations"] - rhs["iterations"])
    return {
        "left": left,
        "right": right,
        "n_common": len(common),
        "left_initial_wrms_better_rate": float(left_better / both_finite) if both_finite else None,
        "initial_wrms_ratio_left_over_right": _numeric_summary(residual_ratios),
        "newton_iteration_delta_left_minus_right": _numeric_summary(iteration_deltas),
        "left_convergence_rate": float(np.mean([left_rows[index]["converged"] for index in common])) if common else None,
        "right_convergence_rate": float(np.mean([right_rows[index]["converged"] for index in common])) if common else None,
    }


def _run_cvode_reference(
    ct,
    mech_path: str,
    phase_name: str | None,
    current_states: np.ndarray,
    target_states: np.ndarray,
    dt: np.ndarray,
    sample_indices: np.ndarray,
    dt_bins: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict:
    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    rows = []
    for local_index, source_index in enumerate(sample_indices):
        current = current_states[local_index]
        target = target_states[local_index]
        setup_started = time.perf_counter()
        gas.TPY = float(current[0]), float(current[1]), current[2:]
        reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
        network = ct.ReactorNet([reactor])
        network.rtol = relative_tolerance
        network.atol = absolute_tolerance
        setup_seconds = time.perf_counter() - setup_started
        advance_started = time.perf_counter()
        network.advance(float(dt[local_index]))
        advance_seconds = time.perf_counter() - advance_started
        phase = reactor.phase if hasattr(reactor, "phase") else reactor.thermo
        predicted = np.concatenate(([float(reactor.T), float(phase.P)], np.asarray(phase.Y, dtype=np.float64)))
        try:
            solver_stats = {
                str(key): float(value)
                for key, value in dict(network.solver_stats).items()
                if np.isscalar(value) and np.isfinite(value)
            }
        except (AttributeError, TypeError, ValueError):
            solver_stats = {}
        rows.append(
            {
                "sample_index": int(source_index),
                "dt_bin": int(dt_bins[local_index]),
                "dt": float(dt[local_index]),
                "setup_seconds": float(setup_seconds),
                "advance_seconds": float(advance_seconds),
                "temperature_abs_error": float(abs(predicted[0] - target[0])),
                "species_mae": float(np.mean(np.abs(predicted[2:] - target[2:]))),
                "solver_stats": solver_stats,
            }
        )

    stats_keys = sorted({key for row in rows for key in row["solver_stats"]})
    by_bin = []
    for bin_value in sorted(np.unique(dt_bins).tolist()):
        local_rows = [row for row in rows if row["dt_bin"] == int(bin_value)]
        by_bin.append(
            {
                "dt_bin": int(bin_value),
                "n_samples": len(local_rows),
                "advance_seconds": _numeric_summary(row["advance_seconds"] for row in local_rows),
                "species_mae": _numeric_summary(row["species_mae"] for row in local_rows),
            }
        )
    return {
        "n_samples": len(rows),
        "setup_seconds": _numeric_summary(row["setup_seconds"] for row in rows),
        "advance_seconds": _numeric_summary(row["advance_seconds"] for row in rows),
        "microseconds_per_advance": float(1e6 * np.mean([row["advance_seconds"] for row in rows])) if rows else None,
        "temperature_abs_error": _numeric_summary(row["temperature_abs_error"] for row in rows),
        "species_mae": _numeric_summary(row["species_mae"] for row in rows),
        "mean_solver_stats": {
            key: float(np.mean([row["solver_stats"][key] for row in rows if key in row["solver_stats"]]))
            for key in stats_keys
        },
        "dt_bin_metrics": by_bin,
        "samples": rows,
    }


def evaluate_implicit_warm_start(
    checkpoint_path: str,
    source_path: str,
    mech_path: str,
    output_path: str,
    *,
    phase_name: str | None = None,
    device: str | None = None,
    max_samples: int = 32,
    sample_seed: int = 20260713,
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
    run_cvode: bool = True,
) -> dict:
    import cantera as ct

    neural_report = evaluate_stoich_interval_model(
        checkpoint_path,
        source_path,
        device=device,
        _return_prediction_arrays=True,
    )
    arrays = neural_report.pop("_prediction_arrays")
    current_all = np.asarray(arrays["current"], dtype=np.float64)
    target_all = np.asarray(arrays["target"], dtype=np.float64)
    prediction_all = np.asarray(arrays["prediction"], dtype=np.float64)
    dt_all = np.asarray(arrays["dt"], dtype=np.float64)
    dt_bin_all = np.asarray(arrays["dt_bin"], dtype=np.int64)
    dt_bin_edges = np.asarray(arrays["dt_bin_edges"], dtype=np.float64)
    species_names = [str(value) for value in arrays["species_names"]]
    attrs = arrays["attrs"]
    phase_name = phase_name or _text_value(attrs.get("phase_name"))

    selected = _stratified_indices(dt_bin_all, max_samples, sample_seed)
    current_states = current_all[selected]
    target_states = target_all[selected]
    neural_states = prediction_all[selected]
    dt_values = dt_all[selected]
    dt_bins = dt_bin_all[selected]

    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    if list(gas.species_names) != species_names:
        raise ValueError(
            "mechanism species order does not match the interval dataset: "
            f"mechanism has {gas.n_species} species, dataset has {len(species_names)}"
        )
    element_matrix = _element_matrix(gas)

    rows = []
    for local_index, source_index in enumerate(selected):
        current_state = current_states[local_index]
        target_state = target_states[local_index]
        neural_state = neural_states[local_index]
        pressure = float(current_state[1])
        current = np.concatenate(([current_state[0]], current_state[2:]))
        target = np.concatenate(([target_state[0]], target_state[2:]))
        neural = np.concatenate(([neural_state[0]], neural_state[2:]))
        rhs = ConstantPressureAdiabaticRHS(gas, pressure, minimum_temperature, positivity_floor)
        explicit_euler = current + float(dt_values[local_index]) * rhs(current)
        neural_positive, limiter_alpha = _positive_segment(
            current,
            neural,
            minimum_temperature,
            positivity_floor,
        )
        guesses = {
            "hold": (current.copy(), None),
            "explicit_euler": (explicit_euler, None),
            "neural": (neural, None),
            "neural_positive_segment": (neural_positive, limiter_alpha),
            "cvode_endpoint": (target.copy(), None),
        }
        absolute_tolerance = np.full(current.size, species_absolute_tolerance, dtype=np.float64)
        absolute_tolerance[0] = temperature_absolute_tolerance
        scale = absolute_tolerance + relative_tolerance * np.abs(current)

        for guess_name, (guess, alpha) in guesses.items():
            initial_diagnostics = _state_diagnostics(guess, current, target, element_matrix)
            solve = _solve_backward_euler(
                rhs,
                current,
                guess,
                float(dt_values[local_index]),
                scale,
                tolerance=newton_tolerance,
                max_iterations=max_newton_iterations,
                max_line_search_steps=max_line_search_steps,
                fd_relative_step=fd_relative_step,
                fd_species_scale=fd_species_scale,
            )
            final_diagnostics = _state_diagnostics(solve["state"], current, target, element_matrix)
            initial_wrms = solve["initial_wrms"]
            one_newton_wrms = solve["one_newton_wrms"]
            rows.append(
                {
                    "sample_index": int(source_index),
                    "dt": float(dt_values[local_index]),
                    "dt_bin": int(dt_bins[local_index]),
                    "guess": guess_name,
                    "positivity_limiter_alpha": alpha,
                    "valid_initial": bool(solve["valid_initial"]),
                    "converged": bool(solve["converged"]),
                    "status": solve["status"],
                    "iterations": int(solve["iterations"]),
                    "initial_wrms": initial_wrms,
                    "one_newton_wrms": one_newton_wrms,
                    "one_newton_reduction_ratio": (
                        float(one_newton_wrms / initial_wrms)
                        if one_newton_wrms is not None and initial_wrms is not None and initial_wrms > 0.0
                        else None
                    ),
                    "final_wrms": solve["final_wrms"],
                    "first_jacobian_condition": solve["first_jacobian_condition"],
                    "rhs_evaluations": int(solve["rhs_evaluations"]),
                    "jacobian_evaluations": int(solve["jacobian_evaluations"]),
                    "linear_solves": int(solve["linear_solves"]),
                    "newton_wall_seconds": float(solve["wall_seconds"]),
                    **{f"initial_{key}": value for key, value in initial_diagnostics.items()},
                    **{f"final_{key}": value for key, value in final_diagnostics.items()},
                }
            )

    guess_names = ["hold", "explicit_euler", "neural", "neural_positive_segment", "cvode_endpoint"]
    guess_metrics = {
        name: _summarize_guess_rows([row for row in rows if row["guess"] == name])
        for name in guess_names
    }
    dt_bin_metrics = []
    for bin_index in sorted(np.unique(dt_bins).tolist()):
        local_rows = [row for row in rows if row["dt_bin"] == int(bin_index)]
        dt_bin_metrics.append(
            {
                "dt_bin": int(bin_index),
                "dt_min": float(dt_bin_edges[bin_index]),
                "dt_max": float(dt_bin_edges[bin_index + 1]),
                "n_samples": int(sum(row["guess"] == "hold" for row in local_rows)),
                "guess_metrics": {
                    name: _summarize_guess_rows([row for row in local_rows if row["guess"] == name])
                    for name in guess_names
                },
            }
        )

    cvode = None
    if run_cvode:
        cvode = _run_cvode_reference(
            ct,
            mech_path,
            phase_name,
            current_states,
            target_states,
            dt_values,
            selected,
            dt_bins,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=species_absolute_tolerance,
        )
        cvode_times = {row["sample_index"]: row["advance_seconds"] for row in cvode["samples"]}
        for row in rows:
            row["cvode_advance_seconds"] = cvode_times.get(row["sample_index"])

    neural_microseconds = neural_report["neural_prediction_runtime"]["microseconds_per_sample"]
    runtime_comparison = {
        "note": "Neural time is batch-amortized; Newton is a Python dense finite-difference diagnostic prototype.",
        "neural_prediction_microseconds_per_sample": float(neural_microseconds),
    }
    if cvode is not None and cvode["microseconds_per_advance"] is not None:
        cvode_microseconds = float(cvode["microseconds_per_advance"])
        runtime_comparison["cvode_advance_microseconds_per_sample"] = cvode_microseconds
        for name in ("hold", "neural", "neural_positive_segment"):
            mean_newton = guess_metrics[name]["newton_wall_seconds"]["mean"]
            if mean_newton is None:
                continue
            model_overhead = neural_microseconds if name.startswith("neural") else 0.0
            hybrid_microseconds = model_overhead + 1e6 * mean_newton
            runtime_comparison[f"{name}_backward_euler_microseconds_per_sample"] = float(hybrid_microseconds)
            runtime_comparison[f"cvode_speedup_over_{name}_backward_euler"] = float(
                cvode_microseconds / hybrid_microseconds
            )

    result = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "mechanism": mech_path,
        "phase_name": phase_name,
        "cantera_version": ct.__version__,
        "device": device,
        "reactor_formulation": "adiabatic_constant_pressure",
        "implicit_method": "backward_euler",
        "state_unknowns": ["temperature", "species_mass_fractions"],
        "pressure_treatment": "fixed_at_interval_initial_pressure",
        "n_source_pairs": int(current_all.shape[0]),
        "n_selected_pairs": int(selected.size),
        "selected_indices": selected.tolist(),
        "species_names": species_names,
        "dt_bin_edges": dt_bin_edges.tolist(),
        "settings": {
            "sample_seed": int(sample_seed),
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
        },
        "neural_interval_metrics": {
            "species_metrics": neural_report["species_metrics"],
            "temperature_mae": neural_report["temperature_mae"],
            "prediction_runtime": neural_report["neural_prediction_runtime"],
        },
        "guess_metrics": guess_metrics,
        "dt_bin_metrics": dt_bin_metrics,
        "comparisons": {
            "neural_vs_hold": _guess_comparison(rows, "neural", "hold"),
            "neural_positive_vs_hold": _guess_comparison(rows, "neural_positive_segment", "hold"),
            "neural_vs_explicit_euler": _guess_comparison(rows, "neural", "explicit_euler"),
            "cvode_endpoint_vs_hold": _guess_comparison(rows, "cvode_endpoint", "hold"),
        },
        "runtime_comparison": runtime_comparison,
        "cvode_reference": cvode,
        "samples": rows,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    csv_path = output.with_name(f"{output.stem}.samples.csv")
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    result["sample_csv"] = str(csv_path)
    return result
