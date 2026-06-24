from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import h5py
import numpy as np

SEQUENCE_SCHEMA_VERSION = "sequence-v2"


@dataclass(frozen=True)
class ReactorCondition:
    temperature: float
    pressure: float
    phi: float
    fuel: str
    oxidizer: str


def fixed_time_grid(*, dt: float, steps: int) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return np.arange(steps + 1, dtype=np.float64) * dt


def log_time_grid(*, t_end: float, steps: int, t_start: float | None = None) -> np.ndarray:
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    first = t_start if t_start is not None else t_end * 1e-8
    if first <= 0 or first >= t_end:
        raise ValueError("t_start must be positive and smaller than t_end")

    grid = np.empty(steps + 1, dtype=np.float64)
    grid[0] = 0.0
    grid[1:] = np.geomspace(first, t_end, steps)
    return grid


def _as_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def build_reactor_conditions(
    *,
    temperatures: Iterable[float],
    pressures: Iterable[float],
    phis: Iterable[float],
    fuel: str,
    oxidizer: str,
) -> list[ReactorCondition]:
    conditions = []
    for temperature in _as_list(temperatures):
        for pressure in _as_list(pressures):
            for phi in _as_list(phis):
                conditions.append(
                    ReactorCondition(
                        temperature=temperature,
                        pressure=pressure,
                        phi=phi,
                        fuel=fuel,
                        oxidizer=oxidizer,
                    )
                )
    return conditions


def generate_constant_pressure_sequence(
    mech_path: str,
    condition: ReactorCondition,
    *,
    phase_name: str | None = None,
    times: Sequence[float] | None = None,
    dt: float | None = None,
    steps: int | None = None,
    energy: str = "on",
) -> np.ndarray:
    import cantera as ct

    if times is None:
        if dt is None or steps is None:
            raise ValueError("Provide either times or both dt and steps")
        times_arr = fixed_time_grid(dt=dt, steps=steps)
    else:
        times_arr = np.asarray(times, dtype=np.float64)
        if times_arr.ndim != 1 or times_arr.shape[0] < 2:
            raise ValueError("times must be a 1D array with at least two entries")
        if times_arr[0] != 0:
            raise ValueError("times must start at 0")
        if np.any(np.diff(times_arr) <= 0):
            raise ValueError("times must be strictly increasing")

    gas = ct.Solution(mech_path, phase_name) if phase_name is not None else ct.Solution(mech_path)
    gas.TP = condition.temperature, condition.pressure
    gas.set_equivalence_ratio(condition.phi, condition.fuel, condition.oxidizer)

    reactor = ct.IdealGasConstPressureReactor(gas, energy=energy)
    network = ct.ReactorNet([reactor])

    sequence = np.empty((times_arr.shape[0], 2 + gas.n_species), dtype=np.float64)
    sequence[0] = np.array([gas.T, gas.P, *gas.Y], dtype=np.float64)

    for step, time_value in enumerate(times_arr[1:], start=1):
        network.advance(float(time_value))
        sequence[step] = np.array([gas.T, gas.P, *gas.Y], dtype=np.float64)

    return sequence


def write_sequence_dataset(
    output_path: str,
    mech_path: str,
    conditions: Sequence[ReactorCondition],
    *,
    phase_name: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    times: Sequence[float] | None = None,
    energy: str = "on",
    mechanism_id: str | None = None,
    sampling: str = "fixed",
    split: str | None = None,
) -> None:
    import cantera as ct

    gas = ct.Solution(mech_path, phase_name) if phase_name is not None else ct.Solution(mech_path)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    if times is None:
        if dt is None or steps is None:
            raise ValueError("Provide either times or both dt and steps")
        times_arr = fixed_time_grid(dt=dt, steps=steps)
    else:
        times_arr = np.asarray(times, dtype=np.float64)

    mech_id = mechanism_id or mech_path

    with h5py.File(output_path, "w") as h5:
        h5.attrs["schema_version"] = SEQUENCE_SCHEMA_VERSION
        h5.attrs["mechanism"] = mech_path
        if phase_name is not None:
            h5.attrs["phase_name"] = phase_name
        h5.attrs["mechanism_id"] = mech_id
        h5.attrs["cantera_version"] = ct.__version__
        h5.attrs["reactor"] = "IdealGasConstPressureReactor"
        h5.attrs["energy"] = energy
        h5.attrs["sampling"] = sampling
        if split is not None:
            h5.attrs["split"] = split
        h5.attrs["steps"] = int(times_arr.shape[0] - 1)
        if sampling == "fixed" and times_arr.shape[0] > 1:
            h5.attrs["dt"] = float(times_arr[1] - times_arr[0])
        h5.create_dataset("species_names", data=np.asarray(gas.species_names, dtype=object), dtype=string_dtype)

        mechanisms = h5.create_group("mechanisms")
        mechanism_group = mechanisms.create_group("000000")
        mechanism_group.attrs["mechanism_id"] = mech_id
        mechanism_group.attrs["mechanism"] = mech_path
        if phase_name is not None:
            mechanism_group.attrs["phase_name"] = phase_name
        mechanism_group.create_dataset("species_names", data=np.asarray(gas.species_names, dtype=object), dtype=string_dtype)

        sequences = h5.create_group("sequences")
        trajectories = h5.create_group("trajectories")
        for idx, condition in enumerate(conditions):
            data = generate_constant_pressure_sequence(
                mech_path,
                condition,
                phase_name=phase_name,
                times=times_arr,
                energy=energy,
            )
            name = f"{idx:06d}"
            dataset = sequences.create_dataset(name, data=data)
            trajectory = trajectories.create_group(name)
            trajectory.create_dataset("states", data=data)
            trajectory.create_dataset("times", data=times_arr)
            for target in (dataset, trajectory):
                target.attrs["mechanism_id"] = mech_id
                target.attrs["mechanism"] = mech_path
                if phase_name is not None:
                    target.attrs["phase_name"] = phase_name
                target.attrs["temperature"] = condition.temperature
                target.attrs["pressure"] = condition.pressure
                target.attrs["phi"] = condition.phi
                target.attrs["fuel"] = condition.fuel
                target.attrs["oxidizer"] = condition.oxidizer
                target.attrs["sampling"] = sampling
                if split is not None:
                    target.attrs["split"] = split


def read_sequence_dataset(input_path: str) -> dict:
    with h5py.File(input_path, "r") as h5:
        if "trajectories" in h5:
            trajectories = {
                name: {
                    "states": group["states"][:],
                    "times": group["times"][:],
                    "attrs": dict(group.attrs),
                }
                for name, group in h5["trajectories"].items()
            }
            sequences = {name: item["states"] for name, item in trajectories.items()}
        else:
            trajectories = {}
            sequences = {
                name: dataset[:]
                for name, dataset in h5["sequences"].items()
            }
        return {
            "attrs": dict(h5.attrs),
            "species_names": [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in h5["species_names"][:]],
            "sequences": sequences,
            "trajectories": trajectories,
        }
