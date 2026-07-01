from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FlameCaseDesign:
    case_id: str
    mechanism: str
    fuel: str
    oxidizer: str
    phi: float
    T_u: float
    P_u: float
    target_lt_over_deltaL: float
    target_uprime_over_Sl: float
    flame_radius: float | None = None
    kernel_temperature: float | None = None
    scalar_initialization_note: str | None = None


@dataclass(frozen=True)
class HitTuningConfig:
    N: int = 128
    L: float = 6.283185307179586
    dt: float = 0.05
    nu: float = 1e-4
    backend: str = "cpu"
    seed: int = 0
    evolve_seconds: float = 0.0
    sample_every: int = 1
    N_values: tuple[int, ...] | None = None
    L_values: tuple[float, ...] | None = None
    k0_values: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0)
    mode_values: tuple[float, ...] | None = None
    bandwidth: float = 1.5
    bandwidth_modes: float = 1.5


def _require_nslab2d() -> dict[str, Any]:
    try:
        from nslab2d.flame_properties import compute_laminar_flame_properties, flame_properties_to_dict
        from nslab2d.openfoam import write_openfoam_u_file
        from nslab2d.field_io import load_velocity_field_npz
        from nslab2d.turbulence_design import (
            compute_regime_targets,
            plot_tuning_search,
            save_tuned_field,
            save_tuning_artifacts,
            tune_field_to_regime_targets,
            tuned_field_result_to_dict,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "NS2dLab_Python is required for 2D HIT flame-case design. "
            "Install DFODE-kit with `pip install -e '.[hit]'`."
        ) from exc

    return {
        "compute_laminar_flame_properties": compute_laminar_flame_properties,
        "flame_properties_to_dict": flame_properties_to_dict,
        "compute_regime_targets": compute_regime_targets,
        "plot_tuning_search": plot_tuning_search,
        "save_tuned_field": save_tuned_field,
        "save_tuning_artifacts": save_tuning_artifacts,
        "tune_field_to_regime_targets": tune_field_to_regime_targets,
        "tuned_field_result_to_dict": tuned_field_result_to_dict,
        "load_velocity_field_npz": load_velocity_field_npz,
        "write_openfoam_u_file": write_openfoam_u_file,
    }


def _achieved_regime_payload(result: Any, targets: Any) -> dict[str, float]:
    uprime = float(result.stats.uprime_component_rms)
    lt = float(result.stats.integral_length_scale_mean)
    S_L = float(targets.S_L)
    delta_L = float(targets.delta_L)
    uprime_over_Sl = uprime / max(S_L, 1e-300)
    lt_over_deltaL = lt / max(delta_L, 1e-300)
    return {
        "uprime": uprime,
        "integral_length_scale": lt,
        "S_L": S_L,
        "delta_L": delta_L,
        "uprime_over_Sl": float(uprime_over_Sl),
        "lt_over_deltaL": float(lt_over_deltaL),
        "Da": float((lt * S_L) / max(uprime * delta_L, 1e-300)),
        "Ka": float((uprime_over_Sl**1.5) * (lt_over_deltaL ** -0.5)),
    }


def design_hit_flame_case(
    design: FlameCaseDesign,
    tuning: HitTuningConfig,
    *,
    output_dir: str | Path,
    S_L: float | None = None,
    delta_L: float | None = None,
    flame_width: float = 0.03,
    transport_model: str = "mixture-averaged",
    loglevel: int = 0,
    write_search_plot: bool = False,
    write_artifacts: bool = False,
) -> dict[str, Any]:
    """Design a mechanism-aware 2D HIT premixed-flame validation case.

    The velocity field is not designed independently from chemistry. The requested
    regime location is interpreted through the mechanism/fuel/phi/T/P laminar flame
    properties, unless `S_L` and `delta_L` are supplied explicitly.
    """
    api = _require_nslab2d()
    output_root = Path(output_dir)
    case_dir = output_root / design.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    flame_payload = None
    if S_L is None or delta_L is None:
        props = api["compute_laminar_flame_properties"](
            mechanism=design.mechanism,
            fuel=design.fuel,
            oxidizer=design.oxidizer,
            phi=design.phi,
            T_u=design.T_u,
            P_u=design.P_u,
            width=flame_width,
            transport_model=transport_model,
            loglevel=loglevel,
        )
        S_L = float(props.S_L)
        delta_L = float(props.delta_alpha)
        flame_payload = api["flame_properties_to_dict"](props)
    else:
        S_L = float(S_L)
        delta_L = float(delta_L)

    targets = api["compute_regime_targets"](
        S_L=S_L,
        delta_L=delta_L,
        lt_over_deltaL=design.target_lt_over_deltaL,
        uprime_over_Sl=design.target_uprime_over_Sl,
    )
    result = api["tune_field_to_regime_targets"](
        targets,
        N=tuning.N,
        L=tuning.L,
        dt=tuning.dt,
        nu=tuning.nu,
        backend_name=tuning.backend,
        seed=tuning.seed,
        evolve_seconds=tuning.evolve_seconds,
        sample_every=tuning.sample_every,
        N_values=list(tuning.N_values) if tuning.N_values is not None else None,
        L_values=list(tuning.L_values) if tuning.L_values is not None else None,
        k0_values=list(tuning.k0_values),
        mode_values=list(tuning.mode_values) if tuning.mode_values is not None else None,
        bandwidth=tuning.bandwidth,
        bandwidth_modes=tuning.bandwidth_modes,
    )

    field_path = case_dir / "velocity_field.npz"
    api["save_tuned_field"](field_path, result, dx=result.best_dx, dy=result.best_dy)

    payload: dict[str, Any] = {
        "case_id": design.case_id,
        "purpose": "mechanism-aware 2D HIT premixed-flame CFD validation initial-condition design",
        "chemistry": {
            "mechanism": design.mechanism,
            "fuel": design.fuel,
            "oxidizer": design.oxidizer,
            "phi": float(design.phi),
            "T_u": float(design.T_u),
            "P_u": float(design.P_u),
            "flame_properties": flame_payload,
        },
        "scalar_initialization": {
            "flame_radius": design.flame_radius,
            "kernel_temperature": design.kernel_temperature,
            "note": design.scalar_initialization_note,
        },
        "requested_regime": asdict(targets),
        "achieved_regime": _achieved_regime_payload(result, targets),
        "hit_tuning": api["tuned_field_result_to_dict"](result),
        "tuning_config": asdict(tuning),
        "outputs": {
            "case_dir": str(case_dir),
            "field_npz": str(field_path),
        },
        "notes": [
            "The HIT velocity field is tuned through flame properties, so the case is mechanism and operating-condition specific.",
            "The velocity field is only one part of the CFD initial condition; downstream DeepFlame/OpenFOAM setup must assign T, Y_i, pressure, and flame/kernel geometry consistently with this case design.",
            "2D HIT regime placement is an approximate validation design tool, not a strict 3D turbulent premixed-flame equivalence.",
        ],
    }

    if write_search_plot:
        search_plot = case_dir / "regime_search.png"
        api["plot_tuning_search"](result, output_path=search_plot, title=f"{design.case_id}: requested vs achieved HIT/flame regime")
        payload["outputs"]["search_plot"] = str(search_plot)

    if write_artifacts:
        artifacts_dir = case_dir / "artifacts"
        payload["outputs"]["artifacts"] = api["save_tuning_artifacts"](result, output_dir=artifacts_dir)

    json_path = case_dir / "case_design.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["outputs"]["case_design_json"] = str(json_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def export_hit_openfoam_u(
    *,
    field_npz: str | Path,
    case_dir: str | Path,
    time_dir: str = "0",
    ordering: str = "x-fastest",
    boundary_json: str | Path | None = None,
) -> Path:
    api = _require_nslab2d()
    field = api["load_velocity_field_npz"](field_npz)
    boundary_patches = None
    if boundary_json is not None:
        boundary_patches = json.loads(Path(boundary_json).read_text(encoding="utf-8"))
    return api["write_openfoam_u_file"](
        field["U"],
        field["V"],
        case_dir=case_dir,
        time_dir=time_dir,
        ordering=ordering,
        boundary_patches=boundary_patches,
    )
