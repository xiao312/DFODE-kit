from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from dfode_kit.data.interval_pairs import load_interval_pair_arrays, load_interval_thermo_arrays
from dfode_kit.evaluation.conservation import conservation_summary
from dfode_kit.evaluation.latent_sequence import _to_jsonable
from dfode_kit.evaluation.metrics import summarize_predictions
from dfode_kit.models.latent_baseline import (
    LatentSubstepStoichiometricIntervalModel,
    StoichiometricIntervalModel,
    ThermoStoichiometricIntervalModel,
)
try:
    from dfode_kit.models.latent_baseline import ThermoProgressSubstepStoichiometricIntervalModel
except ImportError:  # pragma: no cover - allows integration to land before model-owner code.
    ThermoProgressSubstepStoichiometricIntervalModel = None


def _load_model(checkpoint, input_dim: int, device: torch.device):
    cfg = checkpoint.get("training_config", {})
    if checkpoint.get("model_type") == "thermo_stoichiometric_interval":
        model = ThermoStoichiometricIntervalModel(
            input_dim=input_dim,
            stoichiometric_mass_matrix=np.asarray(checkpoint["stoichiometric_mass_matrix"], dtype=np.float64),
            reaction_reversible=np.asarray(checkpoint["reaction_reversible"], dtype=bool),
            latent_dim=int(cfg.get("latent_dim", 16)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            extent_scale=float(cfg.get("thermo_extent_scale", 1e-3)),
            force_scale=float(cfg.get("thermo_force_scale", 1.0)),
            availability_threshold=float(cfg.get("thermo_availability_threshold", -60.0)),
            availability_slope=float(cfg.get("thermo_availability_slope", 1.0)),
        )
    elif checkpoint.get("model_type") == "latent_substep_stoichiometric_interval":
        model_variant = cfg.get("model_variant", "substep")
        model = LatentSubstepStoichiometricIntervalModel(
            input_dim=input_dim,
            stoichiometric_mass_matrix=np.asarray(checkpoint["stoichiometric_mass_matrix"], dtype=np.float64),
            latent_dim=int(cfg.get("latent_dim", 16)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            substeps=int(cfg.get("substeps", 2)),
            flux_mode=cfg.get("flux_mode", "signed-power"),
            transform_alpha=float(cfg.get("transform_alpha", 0.1)),
            transform_scale_by_alpha=bool(cfg.get("transform_scale_by_alpha", True)),
            thermo_input_dim=2 * len(checkpoint["reaction_reversible"]) if model_variant == "substep-soft-thermo" else 0,
            thermo_embedding_dim=int(cfg.get("thermo_embedding_dim", 32)),
            reaction_extent_raw_clip=float(cfg.get("reaction_extent_raw_clip", 0.0)),
        )
    elif checkpoint.get("model_type") == "thermo_progress_substep_stoichiometric_interval":
        if ThermoProgressSubstepStoichiometricIntervalModel is None:
            raise ImportError(
                "thermo_progress_substep_stoichiometric_interval checkpoints require "
                "ThermoProgressSubstepStoichiometricIntervalModel in dfode_kit.models.latent_baseline"
            )
        model = ThermoProgressSubstepStoichiometricIntervalModel(
            input_dim=input_dim,
            stoichiometric_mass_matrix=np.asarray(checkpoint["stoichiometric_mass_matrix"], dtype=np.float64),
            reaction_reversible=np.asarray(checkpoint["reaction_reversible"], dtype=bool),
            latent_dim=int(cfg.get("latent_dim", 16)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            substeps=int(cfg.get("substeps", 2)),
            force_clip=float(cfg.get("thermo_progress_force_clip", 8.0)),
            mobility_scale=float(cfg.get("thermo_progress_mobility_scale", 1.0)),
            progress_mode=cfg.get("thermo_progress_mode", "sinh"),
            latent_update_mode=cfg.get("latent_update_mode", "residual"),
            latent_step_scale=float(cfg.get("latent_step_scale", 1.0)),
            proximal_correction_steps=int(cfg.get("proximal_correction_steps", 0)),
            proximal_correction_lr=float(cfg.get("proximal_correction_lr", 0.1)),
            proximal_free_energy_weight=float(cfg.get("proximal_free_energy_weight", 0.0)),
            proximal_y_floor=float(cfg.get("proximal_y_floor", 1e-300)),
            positivity_safety_factor=float(cfg.get("positivity_safety_factor", 0.0)),
        )
    else:
        model = StoichiometricIntervalModel(
            input_dim=input_dim,
            stoichiometric_mass_matrix=np.asarray(checkpoint["stoichiometric_mass_matrix"], dtype=np.float64),
            latent_dim=int(cfg.get("latent_dim", 16)),
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            flux_mode=cfg.get("flux_mode", "signed-power"),
            transform_alpha=float(cfg.get("transform_alpha", 0.1)),
            transform_scale_by_alpha=bool(cfg.get("transform_scale_by_alpha", True)),
        )
    model.load_state_dict(checkpoint["net"], strict=checkpoint.get("model_type") != "latent_substep_stoichiometric_interval")
    model.to(device)
    model.eval()
    return model


def _bin_summaries(true_species, pred_species, true_state, pred_state, dt, dt_bin, dt_bin_edges, species_names, small_thresholds):
    rows = []
    for bin_index in range(len(dt_bin_edges) - 1):
        mask = dt_bin == bin_index
        if not np.any(mask):
            rows.append(
                {
                    "dt_bin": int(bin_index),
                    "dt_min": float(dt_bin_edges[bin_index]),
                    "dt_max": float(dt_bin_edges[bin_index + 1]),
                    "n_pairs": 0,
                }
            )
            continue
        species = summarize_predictions(
            true_species[mask],
            pred_species[mask],
            species_names=species_names,
            small_thresholds=small_thresholds,
        )["overall"]
        state = summarize_predictions(
            true_state[mask],
            pred_state[mask],
            species_names=["T", "P", *species_names],
            small_thresholds=small_thresholds,
        )["overall"]
        rows.append(
            {
                "dt_bin": int(bin_index),
                "dt_min": float(dt_bin_edges[bin_index]),
                "dt_max": float(dt_bin_edges[bin_index + 1]),
                "n_pairs": int(np.sum(mask)),
                "mean_dt": float(np.mean(dt[mask])),
                "species_overall": species,
                "state_overall": state,
                "temperature_mae": float(np.mean(np.abs(pred_state[mask, 0] - true_state[mask, 0]))),
            }
        )
    return rows


def evaluate_stoich_interval_model(
    checkpoint_path: str,
    source_path: str,
    output_path: str | None = None,
    *,
    mech_path: str | None = None,
    device: str | None = None,
    small_thresholds=(1e-15, 1e-12),
) -> dict:
    current, target, dt, dt_bin, dt_bin_edges, species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
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
    model = _load_model(checkpoint, current.shape[-1], torch_device)
    is_thermo_model = checkpoint.get("model_type") == "thermo_stoichiometric_interval"
    is_substep_model = checkpoint.get("model_type") == "latent_substep_stoichiometric_interval"
    is_thermo_progress_model = checkpoint.get("model_type") == "thermo_progress_substep_stoichiometric_interval"
    is_substep_like_model = is_substep_model or is_thermo_progress_model
    cfg = checkpoint.get("training_config", {})
    uses_soft_thermo = is_substep_model and cfg.get("model_variant") == "substep-soft-thermo"
    if is_thermo_model or uses_soft_thermo or is_thermo_progress_model:
        affinity_hat, log_reactant_activity, _reversible = load_interval_thermo_arrays(source_path, dtype=np.float32)
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
    if is_substep_like_model:
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

    current_tensor = torch.tensor(current_norm, dtype=torch.float32, device=torch_device)
    current_y_tensor = torch.tensor(current[:, 2:], dtype=torch.float64, device=torch_device)
    log_dt_tensor = torch.tensor(log_dt_norm, dtype=torch.float32, device=torch_device)
    dt_tensor = torch.tensor(dt, dtype=torch.float64, device=torch_device)
    log_dt_mean_tensor = torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)
    log_dt_std_tensor = torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)
    if is_thermo_model or uses_soft_thermo or is_thermo_progress_model:
        affinity_tensor = torch.tensor(affinity_hat, dtype=torch.float32, device=torch_device)
        log_activity_tensor = torch.tensor(log_reactant_activity, dtype=torch.float32, device=torch_device)
    if is_substep_like_model and substeps_np is not None:
        log_dt_step_tensor = torch.tensor(log_dt_step_norm, dtype=torch.float32, device=torch_device)
        substeps_tensor = torch.tensor(substeps_np, dtype=torch.long, device=torch_device)

    predictions = []
    predicted_substep_chunks = []
    affinity_work_chunks = []
    free_energy_proxy_chunks = []
    ideal_free_energy_chunks = []
    batch_size = 65536
    with torch.no_grad():
        for start in range(0, current.shape[0], batch_size):
            if is_thermo_model:
                output = model(
                    current_tensor[start : start + batch_size],
                    log_dt_tensor[start : start + batch_size],
                    affinity_tensor[start : start + batch_size],
                    log_activity_tensor[start : start + batch_size],
                    current_species=current_y_tensor[start : start + batch_size],
                )
            elif is_thermo_progress_model:
                stop = start + batch_size
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
                    local_next = torch.empty(
                        (current_tensor[start:stop].shape[0], current.shape[1]),
                        dtype=torch.float64,
                        device=torch_device,
                    )
                    local_affinity_work = torch.empty((local_next.shape[0], 1), dtype=torch.float64, device=torch_device)
                    local_free_energy_proxy = torch.empty((local_next.shape[0], 1), dtype=torch.float64, device=torch_device)
                    local_ideal_free_energy = torch.empty((local_next.shape[0], 1), dtype=torch.float64, device=torch_device)
                    for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                        mask = local_substeps == int(k_value)
                        out = model(
                            current_tensor[start:stop][mask],
                            log_dt_tensor[start:stop][mask],
                            affinity_tensor[start:stop][mask],
                            log_activity_tensor[start:stop][mask],
                            current_species=current_y_tensor[start:stop][mask],
                            log_dt_step=local_log_dt_step[mask],
                            substeps=int(k_value),
                        )
                        local_next[mask] = out["next_state"]
                        local_affinity_work[mask] = out["affinity_work"]
                        local_free_energy_proxy[mask] = out["local_free_energy_proxy_delta"]
                        local_ideal_free_energy[mask] = out["ideal_mixture_free_energy_delta"]
                    predicted_substep_chunks.append(local_substeps.detach().cpu().numpy())
                    output = {
                        "next_state": local_next,
                        "affinity_work": local_affinity_work,
                        "local_free_energy_proxy_delta": local_free_energy_proxy,
                        "ideal_mixture_free_energy_delta": local_ideal_free_energy,
                    }
                elif cfg.get("adaptive_substeps", False):
                    stop = start + batch_size
                    local_size = current_tensor[start:stop].shape[0]
                    local_next = torch.empty((local_size, current.shape[1]), dtype=torch.float64, device=torch_device)
                    local_affinity_work = torch.empty((local_size, 1), dtype=torch.float64, device=torch_device)
                    local_free_energy_proxy = torch.empty((local_size, 1), dtype=torch.float64, device=torch_device)
                    local_ideal_free_energy = torch.empty((local_size, 1), dtype=torch.float64, device=torch_device)
                    local_substeps = substeps_tensor[start:stop]
                    for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                        mask = local_substeps == int(k_value)
                        out = model(
                            current_tensor[start:stop][mask],
                            log_dt_tensor[start:stop][mask],
                            affinity_tensor[start:stop][mask],
                            log_activity_tensor[start:stop][mask],
                            current_species=current_y_tensor[start:stop][mask],
                            log_dt_step=log_dt_step_tensor[start:stop][mask],
                            substeps=int(k_value),
                        )
                        local_next[mask] = out["next_state"]
                        local_affinity_work[mask] = out["affinity_work"]
                        local_free_energy_proxy[mask] = out["local_free_energy_proxy_delta"]
                        local_ideal_free_energy[mask] = out["ideal_mixture_free_energy_delta"]
                    output = {
                        "next_state": local_next,
                        "affinity_work": local_affinity_work,
                        "local_free_energy_proxy_delta": local_free_energy_proxy,
                        "ideal_mixture_free_energy_delta": local_ideal_free_energy,
                    }
                else:
                    output = model(
                        current_tensor[start:stop],
                        log_dt_tensor[start:stop],
                        affinity_tensor[start:stop],
                        log_activity_tensor[start:stop],
                        current_species=current_y_tensor[start:stop],
                        log_dt_step=log_dt_step_tensor[start:stop],
                    )
            elif is_substep_model:
                stop = start + batch_size
                if uses_soft_thermo:
                    thermo_features = torch.cat(
                        [
                            affinity_tensor[start:stop],
                            log_activity_tensor[start:stop],
                        ],
                        dim=-1,
                    )
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
                    local_next = torch.empty(
                        (current_tensor[start:stop].shape[0], current.shape[1]),
                        dtype=torch.float64,
                        device=torch_device,
                    )
                    for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                        mask = local_substeps == int(k_value)
                        local_features = thermo_features[mask] if thermo_features is not None else None
                        out = model(
                            current_tensor[start:stop][mask],
                            log_dt_tensor[start:stop][mask],
                            current_species=current_y_tensor[start:stop][mask],
                            log_dt_step=local_log_dt_step[mask],
                            substeps=int(k_value),
                            thermo_features=local_features,
                        )
                        local_next[mask] = out["next_state"]
                    predicted_substep_chunks.append(local_substeps.detach().cpu().numpy())
                    output = {"next_state": local_next}
                elif cfg.get("adaptive_substeps", False):
                    outputs = []
                    local_size = current_tensor[start:stop].shape[0]
                    local_next = torch.empty((local_size, current.shape[1]), dtype=torch.float64, device=torch_device)
                    local_substeps = substeps_tensor[start:stop]
                    for k_value in torch.unique(local_substeps).detach().cpu().tolist():
                        mask = local_substeps == int(k_value)
                        local_features = thermo_features[mask] if thermo_features is not None else None
                        out = model(
                            current_tensor[start:stop][mask],
                            log_dt_tensor[start:stop][mask],
                            current_species=current_y_tensor[start:stop][mask],
                            log_dt_step=log_dt_step_tensor[start:stop][mask],
                            substeps=int(k_value),
                            thermo_features=local_features,
                        )
                        local_next[mask] = out["next_state"]
                    output = {"next_state": local_next}
                else:
                    output = model(
                        current_tensor[start:stop],
                        log_dt_tensor[start:stop],
                        current_species=current_y_tensor[start:stop],
                        log_dt_step=log_dt_step_tensor[start:stop],
                        thermo_features=thermo_features,
                    )
            else:
                output = model(
                    current_tensor[start : start + batch_size],
                    log_dt_tensor[start : start + batch_size],
                    current_species=current_y_tensor[start : start + batch_size],
                )
            predictions.append(output["next_state"].detach().cpu().numpy())
            if "affinity_work" in output:
                affinity_work_chunks.append(output["affinity_work"].detach().cpu().numpy())
            if "local_free_energy_proxy_delta" in output:
                free_energy_proxy_chunks.append(output["local_free_energy_proxy_delta"].detach().cpu().numpy())
            if "ideal_mixture_free_energy_delta" in output:
                ideal_free_energy_chunks.append(output["ideal_mixture_free_energy_delta"].detach().cpu().numpy())
    pred_raw = np.concatenate(predictions, axis=0)
    pred = np.empty_like(target, dtype=np.float64)
    pred[:, :2] = pred_raw[:, :2] * state_std[:2] + state_mean[:2]
    pred[:, 2:] = pred_raw[:, 2:]

    true_species = target[:, 2:]
    pred_species = pred[:, 2:]
    results = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "n_pairs": int(current.shape[0]),
        "dt_min": float(np.min(dt)),
        "dt_max": float(np.max(dt)),
        "dt_bin_edges": dt_bin_edges.tolist(),
        "species_names": species_names,
        "state_metrics": summarize_predictions(
            target,
            pred,
            species_names=["T", "P", *species_names],
            small_thresholds=small_thresholds,
        ),
        "species_metrics": summarize_predictions(
            true_species,
            pred_species,
            species_names=species_names,
            small_thresholds=small_thresholds,
        ),
        "temperature_mae": float(np.mean(np.abs(pred[:, 0] - target[:, 0]))),
        "pressure_mae": float(np.mean(np.abs(pred[:, 1] - target[:, 1]))),
        "dt_bin_metrics": _bin_summaries(
            true_species,
            pred_species,
            target,
            pred,
            dt,
            dt_bin,
            dt_bin_edges,
            species_names,
            small_thresholds,
        ),
        "checkpoint_training_config": checkpoint.get("training_config", {}),
        "checkpoint_final_metrics": checkpoint.get("final_metrics", {}),
    }
    if predicted_substep_chunks:
        predicted_substeps = np.concatenate(predicted_substep_chunks)
        results["predicted_substep_counts"] = {
            str(int(value)): int(np.sum(predicted_substeps == value))
            for value in sorted(np.unique(predicted_substeps))
        }
    if affinity_work_chunks:
        affinity_work = np.concatenate(affinity_work_chunks, axis=0).reshape(-1)
        free_energy_proxy = np.concatenate(free_energy_proxy_chunks, axis=0).reshape(-1)
        results["thermo_monotonicity_metrics"] = {
            "mean_affinity_work": float(np.mean(affinity_work)),
            "min_affinity_work": float(np.min(affinity_work)),
            "negative_affinity_work_rate": float(np.mean(affinity_work < -1e-12)),
            "mean_local_free_energy_proxy_delta": float(np.mean(free_energy_proxy)),
            "max_local_free_energy_proxy_delta": float(np.max(free_energy_proxy)),
            "positive_free_energy_proxy_delta_rate": float(np.mean(free_energy_proxy > 1e-12)),
        }
    if ideal_free_energy_chunks:
        ideal_free_energy_delta = np.concatenate(ideal_free_energy_chunks, axis=0).reshape(-1)
        results["ideal_mixture_free_energy_metrics"] = {
            "mean_delta": float(np.mean(ideal_free_energy_delta)),
            "min_delta": float(np.min(ideal_free_energy_delta)),
            "max_delta": float(np.max(ideal_free_energy_delta)),
            "positive_delta_rate": float(np.mean(ideal_free_energy_delta > 1e-12)),
        }

    if mech_path is not None:
        import cantera as ct

        phase_name = checkpoint.get("phase_name") or attrs.get("phase_name")
        if isinstance(phase_name, bytes):
            phase_name = phase_name.decode("utf-8")
        gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
        results["conservation_metrics"] = conservation_summary(gas, true_species, pred_species)
        results["update_mass_drift"] = {
            "mean_abs_sum_delta": float(np.mean(np.abs(np.sum(pred_species - current[:, 2:], axis=1)))),
            "max_abs_sum_delta": float(np.max(np.abs(np.sum(pred_species - current[:, 2:], axis=1)))),
        }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_to_jsonable(results), indent=2, sort_keys=True))

    return results
