"""Constant-pressure endpoints and independently integrated changes in FP64."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp


def gas_for(mechanism):
    return ct.Solution(mechanism["file"], mechanism["phase"])


def set_state(gas, state):
    # Do not normalize or clip a saved reference state or the direct-change RHS.
    gas.set_unnormalized_mass_fractions(np.asarray(state["Y"], dtype=np.float64))
    gas.TP = state["T"], state["P"]


def initial_state(mechanism, temperature, config):
    gas = gas_for(mechanism)
    gas.TP = temperature, config["pressure_Pa"]
    gas.set_equivalence_ratio(config["equivalence_ratio"], mechanism["fuel"], config["oxidizer"])
    return {"T": float(gas.T), "P": float(gas.P), "Y": gas.Y.tolist()}


def mechanism_metadata(mechanism):
    gas = gas_for(mechanism)
    path = next((Path(root) / mechanism["file"] for root in ct.get_data_directories()
                 if (Path(root) / mechanism["file"]).is_file()), None)
    if path is None:
        raise FileNotFoundError(mechanism["file"])
    return {"id": mechanism["id"], "file": path.name, "phase": gas.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "species": gas.species_names, "species_count": gas.n_species,
            "reaction_count": gas.n_reactions, "elements": gas.element_names}


def budgets(state, config):
    species = config["species_budget"]
    thermal = config["temperature_budget"]
    return np.r_[thermal["atol"] + thermal["rtol"] * abs(state["T"]),
                 species["atol"] + species["rtol"] * np.abs(state["Y"])]


def conservation(gas, state, delta):
    set_state(gas, state)
    h0 = gas.enthalpy_mass
    y0 = np.asarray(state["Y"])
    y1 = y0 + delta[1:]
    element_matrix = np.array([
        [gas.n_atoms(k, element) * gas.atomic_weights[j] / gas.molecular_weights[k]
         for k in range(gas.n_species)] for j, element in enumerate(gas.element_names)
    ])
    set_state(gas, {"T": state["T"] + delta[0], "P": state["P"], "Y": y1})
    return {"mass_delta_sum": float(np.sum(delta[1:])),
            "element_delta_max": float(np.max(np.abs(element_matrix @ delta[1:]))),
            "enthalpy_relative_drift": float(abs(gas.enthalpy_mass - h0) / max(abs(h0), 1.0)),
            "minimum_mass_fraction": float(y1.min()),
            "negative_species_count": int(np.sum(y1 < 0))}


def endpoint(mechanism, state, interval, rtol, atol, max_step=None):
    start = time.perf_counter()
    gas = gas_for(mechanism)
    set_state(gas, state)
    reactor = ct.IdealGasConstPressureReactor(gas, energy="on", clone=False)
    network = ct.ReactorNet([reactor])
    network.rtol, network.atol = rtol, atol
    network.max_steps = 100000
    if max_step is not None:
        network.max_time_step = max_step
    network.advance(interval)
    delta = np.r_[reactor.T - state["T"], reactor.phase.Y - state["Y"]]
    return {"delta": delta.tolist(), "seconds": time.perf_counter() - start,
            "stats": dict(network.solver_stats), "conservation": conservation(gas, state, delta)}


def direct_rhs(gas, state, scale, deadline=None):
    """Return d(delta / fixed_scale)/dt; no subtraction of endpoint states."""
    y0 = np.asarray(state["Y"])

    def rhs(_time, scaled_delta):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("Pilot wall-time limit reached")
        delta = scale * scaled_delta
        gas.set_unnormalized_mass_fractions(y0 + delta[1:])
        gas.TP = state["T"] + delta[0], state["P"]
        rates = gas.net_production_rates
        dy = rates * gas.molecular_weights / gas.density
        dt = -np.dot(gas.partial_molar_enthalpies, rates) / (gas.density * gas.cp_mass)
        return np.r_[dt, dy] / scale

    return rhs


def direct_increment(mechanism, state, interval, config, rtol, atol, deadline=None):
    start = time.perf_counter()
    gas = gas_for(mechanism)
    scale = budgets(state, config)
    solution = solve_ivp(direct_rhs(gas, state, scale, deadline), (0, interval),
                         np.zeros(scale.size), method="Radau", rtol=rtol, atol=atol,
                         max_step=interval / 10)
    if not solution.success:
        raise RuntimeError(solution.message)
    delta = solution.y[:, -1] * scale
    return {"delta": delta.tolist(), "seconds": time.perf_counter() - start,
            "stats": {"nfev": solution.nfev, "njev": solution.njev,
                      "nlu": solution.nlu, "steps": len(solution.t) - 1},
            "conservation": conservation(gas, state, delta)}


def quantized_state(state):
    raw_y = np.asarray(state["Y"], dtype=np.float32).astype(np.float64)
    # A valid composition is required for the physical re-integration test.
    normalized_y = raw_y / np.sum(raw_y)
    result = {"T": float(np.float32(state["T"])), "P": float(np.float32(state["P"])),
              "Y": normalized_y.tolist()}
    effects = {"raw_Y_error_max": float(np.max(np.abs(raw_y - state["Y"]))),
               "normalization_change_max": float(np.max(np.abs(normalized_y - raw_y))),
               "raw_sum_error": float(np.sum(raw_y) - 1),
               "T_error_K": result["T"] - state["T"],
               "P_error_Pa": result["P"] - state["P"]}
    return result, effects


def parent_trajectory(mechanism, temperature, config):
    state = initial_state(mechanism, temperature, config)
    gas = gas_for(mechanism)
    set_state(gas, state)
    reactor = ct.IdealGasConstPressureReactor(gas, energy="on", clone=False)
    network = ct.ReactorNet([reactor])
    tight = config["tolerances"][-1]
    network.rtol, network.atol = tight["rtol"], tight["atol"]
    network.max_steps = 100000
    times = np.r_[0.0, np.geomspace(1e-8, mechanism["duration_s"], 128)]
    samples = []
    for sample_time in times:
        if sample_time:
            network.advance(float(sample_time))
        samples.append({"time_s": float(sample_time), "T": float(reactor.T),
                        "P": float(reactor.phase.P), "Y": reactor.phase.Y.tolist()})
    temperatures = np.array([sample["T"] for sample in samples])
    peak = int(np.argmax(np.diff(temperatures))) + 1
    selected = {0, 32, 64, 96, 128, max(1, peak - 1), peak, min(128, peak + 1)}
    # Fill duplicates deterministically; never exceed the specified anchor count.
    for index in range(1, 129):
        if len(selected) >= config["anchors_per_trajectory"]:
            break
        selected.add(index)
    indices = sorted(selected)[:config["anchors_per_trajectory"]]
    return {"initial_state": state, "scout_times_s": times.tolist(),
            "scout_temperatures_K": temperatures.tolist(), "anchor_indices": indices,
            "anchors": [samples[index] for index in indices],
            "selection": "five fixed time indices plus three around greatest sampled temperature rise; fill duplicates in time order",
            "stats": dict(network.solver_stats)}
