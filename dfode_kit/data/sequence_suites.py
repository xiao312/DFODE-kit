from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json
import time

import numpy as np

from dfode_kit.data.sequences import (
    ReactorCondition,
    build_reactor_conditions,
    fixed_time_grid,
    log_time_grid,
    write_sequence_dataset,
)
from dfode_kit.data.reactivity import summarize_sequence_file_reactivity

ATM = 101325.0


@dataclass(frozen=True)
class SequenceCase:
    case_id: str
    mechanism: str
    mechanism_id: str
    fuel: str
    oxidizer: str
    temperatures: tuple[float, ...]
    pressures: tuple[float, ...]
    phis: tuple[float, ...]
    sampling: str
    steps: int
    phase_name: str | None = None
    dt: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    energy: str = "on"


def _tuple(values) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def suite_cases(profile: str = "pilot") -> list[SequenceCase]:
    if profile == "pilot":
        return [
            SequenceCase(
                case_id="burke_h2_logtime",
                mechanism="mechanisms/Burke2012_s9r23.yaml",
                mechanism_id="burke_h2_9s23r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([900, 1100, 1300, 1500]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.6, 1.0, 1.4]),
                sampling="log-time",
                t_start=1e-9,
                t_end=1e-5,
                steps=64,
            ),
            SequenceCase(
                case_id="h2o2_h2_logtime",
                mechanism="h2o2.yaml",
                mechanism_id="cantera_h2o2_10s29r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([900, 1100, 1300, 1500]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.6, 1.0, 1.4]),
                sampling="log-time",
                t_start=1e-9,
                t_end=1e-5,
                steps=64,
            ),
            SequenceCase(
                case_id="gri30_ch4_logtime",
                mechanism="gri30.yaml",
                mechanism_id="gri30_ch4_53s325r",
                fuel="CH4:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1000, 1300, 1600]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-4,
                steps=64,
            ),
        ]

    if profile == "small":
        return [
            replace(
                case,
                temperatures=_tuple([900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800])
                if "h2" in case.case_id
                else _tuple([1000, 1200, 1400, 1600, 1800]),
                pressures=_tuple([ATM, 5 * ATM, 10 * ATM]),
                phis=_tuple([0.6, 0.8, 1.0, 1.2, 1.4]),
                steps=128,
            )
            for case in suite_cases("pilot")
        ]

    if profile == "reactive-pilot":
        return [
            SequenceCase(
                case_id="burke_h2_reactive_logtime",
                mechanism="mechanisms/Burke2012_s9r23.yaml",
                mechanism_id="burke_h2_9s23r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1200, 1400, 1600, 1800, 2000]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.6, 1.0, 1.4]),
                sampling="log-time",
                t_start=1e-10,
                t_end=1e-4,
                steps=128,
            ),
            SequenceCase(
                case_id="h2o2_h2_reactive_logtime",
                mechanism="h2o2.yaml",
                mechanism_id="cantera_h2o2_10s29r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1200, 1400, 1600, 1800, 2000]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.6, 1.0, 1.4]),
                sampling="log-time",
                t_start=1e-10,
                t_end=1e-4,
                steps=128,
            ),
            SequenceCase(
                case_id="gri30_ch4_reactive_logtime",
                mechanism="gri30.yaml",
                mechanism_id="gri30_ch4_53s325r",
                fuel="CH4:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1400, 1600, 1800, 2000]),
                pressures=_tuple([ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-2,
                steps=128,
            ),
        ]

    if profile == "crossfuel-packaged-pilot":
        return [
            SequenceCase(
                case_id="h2o2_h2_crossfuel_logtime",
                mechanism="h2o2.yaml",
                mechanism_id="cantera_h2o2_10s29r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1200, 1600, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-10,
                t_end=1e-4,
                steps=128,
            ),
            SequenceCase(
                case_id="ohn_h2_crossfuel_logtime",
                mechanism="ohn.yaml",
                mechanism_id="cantera_ohn_h2_18s69r",
                fuel="H2:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1200, 1600, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-10,
                t_end=1e-4,
                steps=128,
            ),
            SequenceCase(
                case_id="ohn_nh3_crossfuel_logtime",
                mechanism="ohn.yaml",
                mechanism_id="cantera_ohn_nh3_18s69r",
                fuel="NH3:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1700, 2000, 2300]),
                pressures=_tuple([5 * ATM, 10 * ATM]),
                phis=_tuple([0.8, 1.0, 1.2]),
                sampling="log-time",
                t_start=1e-9,
                t_end=5e-2,
                steps=128,
            ),
            SequenceCase(
                case_id="ptcombust_ch4_crossfuel_logtime",
                mechanism="ptcombust.yaml",
                mechanism_id="cantera_ptcombust_ch4_32s186r",
                fuel="CH4:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1400, 1700, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-2,
                steps=128,
            ),
            SequenceCase(
                case_id="ptcombust_c2h6_crossfuel_logtime",
                mechanism="ptcombust.yaml",
                mechanism_id="cantera_ptcombust_c2h6_32s186r",
                fuel="C2H6:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1400, 1700, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-2,
                steps=128,
            ),
            SequenceCase(
                case_id="gri30_ch4_crossfuel_logtime",
                mechanism="gri30.yaml",
                mechanism_id="gri30_ch4_53s325r",
                fuel="CH4:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1400, 1700, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-2,
                steps=128,
            ),
            SequenceCase(
                case_id="gri30_c3h8_crossfuel_logtime",
                mechanism="gri30.yaml",
                mechanism_id="gri30_c3h8_53s325r",
                fuel="C3H8:1.0",
                oxidizer="O2:1.0,N2:3.76",
                temperatures=_tuple([1400, 1700, 2000]),
                pressures=_tuple([ATM, 5 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=1e-2,
                steps=128,
            ),
            SequenceCase(
                case_id="ndodecane_c12h26_crossfuel_logtime",
                mechanism="nDodecane_Reitz.yaml",
                mechanism_id="cantera_ndodecane_reitz_100s553r",
                phase_name="nDodecane_IG",
                fuel="c12h26:1.0",
                oxidizer="o2:1.0,n2:3.76",
                temperatures=_tuple([1000, 1250, 1500]),
                pressures=_tuple([10 * ATM, 20 * ATM]),
                phis=_tuple([0.7, 1.0, 1.3]),
                sampling="log-time",
                t_start=1e-8,
                t_end=2e-2,
                steps=128,
            ),
        ]

    raise ValueError(
        f"Unknown suite profile '{profile}'. Expected 'pilot', 'small', 'reactive-pilot', "
        "or 'crossfuel-packaged-pilot'."
    )


def split_conditions(
    conditions: list[ReactorCondition],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[ReactorCondition], list[ReactorCondition]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(conditions) < 2:
        raise ValueError("At least two conditions are required for train/validation split")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(conditions))
    rng.shuffle(indices)

    n_val = max(1, int(round(len(conditions) * val_fraction)))
    n_val = min(n_val, len(conditions) - 1)
    val_indices = set(int(idx) for idx in indices[:n_val])

    train = [condition for idx, condition in enumerate(conditions) if idx not in val_indices]
    val = [condition for idx, condition in enumerate(conditions) if idx in val_indices]
    return train, val


def _case_times(case: SequenceCase) -> np.ndarray:
    if case.sampling == "fixed":
        if case.dt is None:
            raise ValueError(f"Case {case.case_id} uses fixed sampling but dt is missing")
        return fixed_time_grid(dt=case.dt, steps=case.steps)
    if case.sampling == "log-time":
        if case.t_end is None:
            raise ValueError(f"Case {case.case_id} uses log-time sampling but t_end is missing")
        return log_time_grid(t_end=case.t_end, t_start=case.t_start, steps=case.steps)
    raise ValueError(f"Unsupported sampling strategy '{case.sampling}'")


def generate_suite(
    output_dir: str,
    *,
    profile: str = "pilot",
    val_fraction: float = 0.2,
    seed: int = 260624,
    only: set[str] | None = None,
    diagnose_reactivity: bool = True,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "dfode-sequence-suite-v1",
        "profile": profile,
        "seed": seed,
        "val_fraction": val_fraction,
        "created_unix": time.time(),
        "cases": [],
    }

    for case in suite_cases(profile):
        if only and case.case_id not in only and case.mechanism_id not in only:
            continue

        conditions = build_reactor_conditions(
            temperatures=case.temperatures,
            pressures=case.pressures,
            phis=case.phis,
            fuel=case.fuel,
            oxidizer=case.oxidizer,
        )
        train_conditions, val_conditions = split_conditions(
            conditions,
            val_fraction=val_fraction,
            seed=seed + len(manifest["cases"]),
        )
        times = _case_times(case)

        case_record = {
            **asdict(case),
            "n_conditions": len(conditions),
            "n_train": len(train_conditions),
            "n_val": len(val_conditions),
            "splits": {},
        }

        for split_name, split_conditions_for_case in (
            ("train", train_conditions),
            ("val", val_conditions),
        ):
            path = out_dir / f"{case.case_id}_{split_name}.h5"
            write_sequence_dataset(
                str(path),
                case.mechanism,
                split_conditions_for_case,
                phase_name=case.phase_name,
                times=times,
                energy=case.energy,
                mechanism_id=case.mechanism_id,
                sampling=case.sampling,
                split=split_name,
            )
            reactivity = None
            if diagnose_reactivity:
                reactivity = summarize_sequence_file_reactivity(str(path))["aggregate"]
            case_record["splits"][split_name] = {
                "path": str(path),
                "n_conditions": len(split_conditions_for_case),
                "n_steps": int(times.shape[0] - 1),
                "t_end": float(times[-1]),
                "reactivity": reactivity,
            }

        manifest["cases"].append(case_record)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
