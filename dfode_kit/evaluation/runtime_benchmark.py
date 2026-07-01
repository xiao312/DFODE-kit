from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from dfode_kit.data.interval_pairs import load_interval_pair_arrays, load_interval_thermo_arrays
from dfode_kit.evaluation.latent_sequence import _to_jsonable
from dfode_kit.evaluation.stoich_interval import _load_model


def benchmark_stoich_interval_runtime(
    checkpoint_path: str,
    source_path: str,
    output_path: str | None = None,
    *,
    device: str | None = None,
    batch_size: int = 65536,
    repeat: int = 20,
    warmup: int = 5,
    max_samples: int | None = None,
    mech_path: str | None = None,
    cvode_samples: int = 0,
) -> dict:
    current, _target, dt, _dt_bin, _dt_bin_edges, _species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
    if max_samples is not None:
        current = current[:max_samples]
        dt = dt[:max_samples]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    log_dt_mean = np.asarray(checkpoint["log_dt_mean"], dtype=np.float64)
    log_dt_std = np.asarray(checkpoint["log_dt_std"], dtype=np.float64)
    state_std = np.where(state_std > 0, state_std, 1.0)
    log_dt_std = np.where(log_dt_std > 0, log_dt_std, 1.0)
    current_norm = ((current - state_mean) / state_std).astype(np.float32)
    log_dt = np.log(np.maximum(dt, 1e-300))[:, None]
    log_dt_norm = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)

    cfg = checkpoint.get("training_config", {})
    model_type = checkpoint.get("model_type")
    is_thermo = model_type == "thermo_stoichiometric_interval"
    is_substep = model_type == "latent_substep_stoichiometric_interval"
    is_thermo_progress = model_type == "thermo_progress_substep_stoichiometric_interval"
    is_substep_like = is_substep or is_thermo_progress
    uses_soft_thermo = is_substep and cfg.get("model_variant") == "substep-soft-thermo"
    if is_thermo or uses_soft_thermo or is_thermo_progress:
        affinity_hat, log_reactant_activity, _reversible = load_interval_thermo_arrays(source_path, dtype=np.float32)
        if max_samples is not None:
            affinity_hat = affinity_hat[:max_samples]
            log_reactant_activity = log_reactant_activity[:max_samples]
        if uses_soft_thermo:
            thermo_feature_mean = checkpoint.get("thermo_feature_mean")
            thermo_feature_std = checkpoint.get("thermo_feature_std")
            if thermo_feature_mean is not None and thermo_feature_std is not None:
                thermo_features = np.concatenate([affinity_hat, log_reactant_activity], axis=1).astype(np.float64)
                thermo_features = (thermo_features - np.asarray(thermo_feature_mean, dtype=np.float64)) / np.asarray(thermo_feature_std, dtype=np.float64)
                clip_value = float(cfg.get("thermo_feature_clip", 10.0))
                if clip_value > 0.0:
                    thermo_features = np.clip(thermo_features, -clip_value, clip_value)
                n_reactions = affinity_hat.shape[1]
                affinity_hat = thermo_features[:, :n_reactions].astype(np.float32)
                log_reactant_activity = thermo_features[:, n_reactions:].astype(np.float32)
    if is_substep_like:
        if cfg.get("learned_adaptive_substeps", False):
            substeps_np = None
            log_dt_step_norm = None
        elif cfg.get("adaptive_substeps", False):
            substeps_np = np.ones_like(dt, dtype=np.int64)
            substeps_np = np.where(dt >= 1e-7, 2, substeps_np)
            substeps_np = np.where(dt >= 1e-5, 4, substeps_np)
            substeps_np = np.where(dt >= 1e-4, 8, substeps_np)
        else:
            substeps_np = np.full_like(dt, int(cfg.get("substeps", 2)), dtype=np.int64)
        if substeps_np is not None:
            log_dt_step = np.log(np.maximum(dt / substeps_np, 1e-300))[:, None]
            log_dt_step_norm = ((log_dt_step - log_dt_mean) / log_dt_std).astype(np.float32)

    model = _load_model(checkpoint, current.shape[-1], torch_device)
    current_tensor = torch.tensor(current_norm, dtype=torch.float32, device=torch_device)
    current_y_tensor = torch.tensor(current[:, 2:], dtype=torch.float64, device=torch_device)
    log_dt_tensor = torch.tensor(log_dt_norm, dtype=torch.float32, device=torch_device)
    dt_tensor = torch.tensor(dt, dtype=torch.float64, device=torch_device)
    log_dt_mean_tensor = torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)
    log_dt_std_tensor = torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)
    if is_thermo or uses_soft_thermo or is_thermo_progress:
        affinity_tensor = torch.tensor(affinity_hat, dtype=torch.float32, device=torch_device)
        log_activity_tensor = torch.tensor(log_reactant_activity, dtype=torch.float32, device=torch_device)
    if is_substep_like and substeps_np is not None:
        log_dt_step_tensor = torch.tensor(log_dt_step_norm, dtype=torch.float32, device=torch_device)
        substeps_tensor = torch.tensor(substeps_np, dtype=torch.long, device=torch_device)

    def run_once():
        with torch.no_grad():
            for start in range(0, current.shape[0], batch_size):
                stop = start + batch_size
                if is_thermo:
                    model(
                        current_tensor[start:stop],
                        log_dt_tensor[start:stop],
                        affinity_tensor[start:stop],
                        log_activity_tensor[start:stop],
                        current_species=current_y_tensor[start:stop],
                    )
                elif is_thermo_progress:
                    if cfg.get("learned_adaptive_substeps", False):
                        local_substeps, _policy_logits = model.predict_substeps(
                            current_tensor[start:stop],
                            log_dt_tensor[start:stop],
                            affinity_tensor[start:stop],
                            log_activity_tensor[start:stop],
                        )
                        local_log_dt_step = (
                            (
                                torch.log(torch.clamp(dt_tensor[start:stop] / local_substeps.to(torch.float64), min=1e-300))[:, None]
                                - log_dt_mean_tensor
                            )
                            / log_dt_std_tensor
                        ).to(torch.float32)
                        for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                            mask = local_substeps == int(k_value)
                            model(
                                current_tensor[start:stop][mask],
                                log_dt_tensor[start:stop][mask],
                                affinity_tensor[start:stop][mask],
                                log_activity_tensor[start:stop][mask],
                                current_species=current_y_tensor[start:stop][mask],
                                log_dt_step=local_log_dt_step[mask],
                                substeps=int(k_value),
                            )
                    elif cfg.get("adaptive_substeps", False):
                        local_substeps = substeps_tensor[start:stop]
                        for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                            mask = local_substeps == int(k_value)
                            model(
                                current_tensor[start:stop][mask],
                                log_dt_tensor[start:stop][mask],
                                affinity_tensor[start:stop][mask],
                                log_activity_tensor[start:stop][mask],
                                current_species=current_y_tensor[start:stop][mask],
                                log_dt_step=log_dt_step_tensor[start:stop][mask],
                                substeps=int(k_value),
                            )
                    else:
                        model(
                            current_tensor[start:stop],
                            log_dt_tensor[start:stop],
                            affinity_tensor[start:stop],
                            log_activity_tensor[start:stop],
                            current_species=current_y_tensor[start:stop],
                            log_dt_step=log_dt_step_tensor[start:stop],
                        )
                elif is_substep:
                    if uses_soft_thermo:
                        thermo_features = torch.cat([affinity_tensor[start:stop], log_activity_tensor[start:stop]], dim=-1)
                    else:
                        thermo_features = None
                    if cfg.get("learned_adaptive_substeps", False):
                        local_substeps, _policy_logits = model.predict_substeps(
                            current_tensor[start:stop],
                            log_dt_tensor[start:stop],
                            thermo_features=thermo_features,
                        )
                        local_log_dt_step = (
                            (
                                torch.log(torch.clamp(dt_tensor[start:stop] / local_substeps.to(torch.float64), min=1e-300))[:, None]
                                - log_dt_mean_tensor
                            )
                            / log_dt_std_tensor
                        ).to(torch.float32)
                        for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                            mask = local_substeps == int(k_value)
                            local_features = thermo_features[mask] if thermo_features is not None else None
                            model(
                                current_tensor[start:stop][mask],
                                log_dt_tensor[start:stop][mask],
                                current_species=current_y_tensor[start:stop][mask],
                                log_dt_step=local_log_dt_step[mask],
                                substeps=int(k_value),
                                thermo_features=local_features,
                            )
                    elif cfg.get("adaptive_substeps", False):
                        local_substeps = substeps_tensor[start:stop]
                        for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                            mask = local_substeps == int(k_value)
                            local_features = thermo_features[mask] if thermo_features is not None else None
                            model(
                                current_tensor[start:stop][mask],
                                log_dt_tensor[start:stop][mask],
                                current_species=current_y_tensor[start:stop][mask],
                                log_dt_step=log_dt_step_tensor[start:stop][mask],
                                substeps=int(k_value),
                                thermo_features=local_features,
                            )
                    else:
                        model(
                            current_tensor[start:stop],
                            log_dt_tensor[start:stop],
                            current_species=current_y_tensor[start:stop],
                            log_dt_step=log_dt_step_tensor[start:stop],
                            thermo_features=thermo_features,
                        )
                else:
                    model(
                        current_tensor[start:stop],
                        log_dt_tensor[start:stop],
                        current_species=current_y_tensor[start:stop],
                    )

    for _idx in range(warmup):
        run_once()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    timings = []
    for _idx in range(repeat):
        start_time = time.perf_counter()
        run_once()
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        timings.append(time.perf_counter() - start_time)

    result = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "device": str(torch_device),
        "model_type": model_type,
        "training_config": cfg,
        "n_samples": int(current.shape[0]),
        "batch_size": int(batch_size),
        "repeat": int(repeat),
        "warmup": int(warmup),
        "mean_seconds": float(np.mean(timings)),
        "std_seconds": float(np.std(timings)),
        "min_seconds": float(np.min(timings)),
        "max_seconds": float(np.max(timings)),
        "microseconds_per_sample": float(1e6 * np.mean(timings) / current.shape[0]),
    }
    if mech_path is not None and cvode_samples > 0:
        import cantera as ct

        phase_name = checkpoint.get("phase_name") or attrs.get("phase_name")
        if isinstance(phase_name, bytes):
            phase_name = phase_name.decode("utf-8")
        gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
        n_cvode = min(int(cvode_samples), current.shape[0])
        cvode_timings = []
        for idx in range(n_cvode):
            gas.TPY = float(current[idx, 0]), float(current[idx, 1]), current[idx, 2:]
            reactor = ct.IdealGasConstPressureReactor(gas, clone=False)
            network = ct.ReactorNet([reactor])
            start_time = time.perf_counter()
            network.advance(float(dt[idx]))
            cvode_timings.append(time.perf_counter() - start_time)
        result["cvode"] = {
            "n_samples": int(n_cvode),
            "mean_seconds": float(np.mean(cvode_timings)),
            "std_seconds": float(np.std(cvode_timings)),
            "min_seconds": float(np.min(cvode_timings)),
            "max_seconds": float(np.max(cvode_timings)),
            "microseconds_per_sample": float(1e6 * np.mean(cvode_timings)),
            "neural_speedup_vs_cvode": float(np.mean(cvode_timings) / (np.mean(timings) / current.shape[0])),
        }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
    return result
