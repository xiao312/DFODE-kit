"""Fixed-temperature, fixed-volume chemistry matching the inspected CFD runtime.

Inputs are validated mass fractions. No state clipping or normalization happens
here. The independent increment solve shares Cantera rates, but not CVODE's
endpoint formulation. Agreement is evidence, not a rigorous accuracy bound.
"""
from __future__ import annotations

import time

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp


def validate_state(gas, state):
    temperature, pressure = float(state["T"]), float(state["P"])
    fractions = np.asarray(state["Y"], dtype=np.float64)
    if not np.isfinite([temperature, pressure]).all() or min(temperature, pressure) <= 0:
        raise ValueError("Temperature and pressure must be positive finite values")
    if fractions.shape != (gas.n_species,) or not np.isfinite(fractions).all():
        raise ValueError("Y must be a finite vector in mechanism species order")
    if np.any(fractions < 0) or abs(float(fractions.sum()) - 1) > 1e-10:
        raise ValueError("Y must be nonnegative and sum to one within 1e-10")
    return temperature, pressure, fractions


def set_state(gas, state):
    temperature, pressure, fractions = validate_state(gas, state)
    gas.set_unnormalized_mass_fractions(fractions)
    gas.TP = temperature, pressure


def element_matrix(gas):
    return np.array([
        [gas.n_atoms(k, j) * gas.atomic_weights[j] / gas.molecular_weights[k]
         for k in range(gas.n_species)] for j in range(gas.n_elements)
    ])


def diagnostics(gas, state, delta, density_initial):
    fractions = np.asarray(state["Y"]) + delta
    return {
        "mass_delta_sum": float(np.sum(delta)),
        "element_delta_max": float(np.max(np.abs(element_matrix(gas) @ delta))),
        "minimum_mass_fraction": float(fractions.min()),
        "negative_species_count": int(np.sum(fractions < 0)),
        "temperature_change_K": float(gas.T - state["T"]),
        "pressure_change_Pa": float(gas.P - state["P"]),
        "relative_density_change": float((gas.density - density_initial) / density_initial),
    }


def validate_solve(interval, rtol, atol):
    if not np.isfinite([interval, rtol, atol]).all() or min(interval, rtol, atol) <= 0:
        raise ValueError("Interval and tolerances must be positive and finite")


def endpoint(mechanism, state, interval=1e-6, rtol=1e-12, atol=1e-18,
             max_step=None):
    validate_solve(interval, rtol, atol)
    if max_step is not None and (not np.isfinite(max_step) or max_step <= 0):
        raise ValueError("max_step must be positive and finite")
    started = time.perf_counter()
    gas = ct.Solution(str(mechanism))
    set_state(gas, state)
    density_initial = gas.density
    reactor = ct.IdealGasReactor(gas, energy="off", clone=False)
    network = ct.ReactorNet([reactor])
    network.rtol, network.atol = rtol, atol
    network.max_steps = 100000
    if max_step is not None:
        network.max_time_step = max_step
    network.advance(interval)
    delta = reactor.phase.Y - np.asarray(state["Y"])
    return {
        "delta": delta.tolist(), "seconds": time.perf_counter() - started,
        "stats": dict(network.solver_stats),
        "diagnostics": diagnostics(reactor.phase, state, delta, density_initial),
    }


def direct_rhs(gas, state, scale, deadline=None):
    set_state(gas, state)
    density_initial = float(gas.density)
    fractions_initial = np.asarray(state["Y"], dtype=np.float64).copy()
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != fractions_initial.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scale must contain one positive finite value per species")

    def rhs(_time, scaled_delta):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Direct increment audit reached its deadline")
        # Keep rho, not p, fixed. Do not normalize intermediate implicit-solver
        # trial states: normalization would change the differential equation.
        gas.set_unnormalized_mass_fractions(fractions_initial + scale * scaled_delta)
        gas.TD = state["T"], density_initial
        return gas.net_production_rates * gas.molecular_weights / density_initial / scale

    return rhs, density_initial


def direct_increment(mechanism, state, interval=1e-6, rtol=1e-10, atol=1e-6,
                     deadline=None):
    validate_solve(interval, rtol, atol)
    started = time.perf_counter()
    gas = ct.Solution(str(mechanism))
    scale = 1e-12 + 1e-6 * np.abs(np.asarray(state["Y"], dtype=np.float64))
    rhs, density_initial = direct_rhs(gas, state, scale, deadline)
    solution = solve_ivp(rhs, (0, interval), np.zeros(gas.n_species), method="Radau",
                         rtol=rtol, atol=atol, max_step=interval / 10)
    if not solution.success:
        raise RuntimeError(solution.message)
    delta = solution.y[:, -1] * scale
    gas.set_unnormalized_mass_fractions(np.asarray(state["Y"]) + delta)
    gas.TD = state["T"], density_initial
    return {
        "delta": delta.tolist(), "seconds": time.perf_counter() - started,
        "stats": {"nfev": solution.nfev, "njev": solution.njev,
                  "nlu": solution.nlu, "steps": len(solution.t) - 1},
        "diagnostics": diagnostics(gas, state, delta, density_initial),
    }
