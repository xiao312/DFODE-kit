from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import h5py

from dfode_kit.evaluation.conservation import conservation_summary
from dfode_kit.evaluation.metrics import summarize_predictions
from dfode_kit.models.latent_baseline import AutoencoderLatentGRU, ConservedLatentDeltaModel, StoichiometricFluxModel
from dfode_kit.training.latent_sequence import load_sequence_arrays


def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_model(checkpoint_path: str, input_dim: int, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("training_config", {})
    model = AutoencoderLatentGRU(
        input_dim=input_dim,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("num_layers", 1)),
    )
    model.load_state_dict(checkpoint["net"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _load_conserved_model(checkpoint, input_dim: int, device: torch.device):
    cfg = checkpoint.get("training_config", {})
    model = ConservedLatentDeltaModel(
        input_dim=input_dim,
        completion_matrix=np.asarray(checkpoint["completion_matrix"], dtype=np.float64),
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        key_mode=cfg.get("key_mode", "direct"),
        transform_alpha=float(cfg.get("transform_alpha", 0.5)),
        transform_eps=float(cfg.get("transform_eps", 1e-12)),
        transform_scale_by_alpha=bool(cfg.get("transform_scale_by_alpha", False)),
        key_species_indices=checkpoint.get("key_species_indices"),
        key_scale_alpha=float(cfg.get("key_scale_alpha", 0.0)),
    )
    model.load_state_dict(checkpoint["net"])
    model.to(device)
    model.eval()
    return model


def _load_stoich_model(checkpoint, input_dim: int, device: torch.device):
    cfg = checkpoint.get("training_config", {})
    model = StoichiometricFluxModel(
        input_dim=input_dim,
        stoichiometric_mass_matrix=np.asarray(checkpoint["stoichiometric_mass_matrix"], dtype=np.float64),
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        flux_mode=cfg.get("flux_mode", "direct"),
        transform_alpha=float(cfg.get("transform_alpha", 0.1)),
        transform_scale_by_alpha=bool(cfg.get("transform_scale_by_alpha", True)),
    )
    model.load_state_dict(checkpoint["net"])
    model.to(device)
    model.eval()
    return model


def _normalization_from_checkpoint(checkpoint, input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if mean.shape != (input_dim,) or std.shape != (input_dim,):
        raise ValueError(
            f"Checkpoint normalization shape mismatch: mean={mean.shape}, std={std.shape}, input_dim={input_dim}"
        )
    std = np.where(std > 0, std, 1.0)
    return mean, std


def _relative_time_error(true_dt: np.ndarray, pred_dt: np.ndarray) -> dict[str, float]:
    rel = np.abs(pred_dt - true_dt) / np.maximum(np.abs(true_dt), 1e-300)
    return {
        "mae_log_dt": float(np.mean(np.abs(np.log(np.maximum(pred_dt, 1e-300)) - np.log(np.maximum(true_dt, 1e-300))))),
        "mae_dt": float(np.mean(np.abs(pred_dt - true_dt))),
        "median_relative_dt_error": float(np.median(rel)),
        "mean_relative_dt_error": float(np.mean(rel)),
    }


def _read_sequence_phase_name(source_path: str) -> str | None:
    with h5py.File(source_path, "r") as h5:
        phase = h5.attrs.get("phase_name")
    if phase is None:
        return None
    if isinstance(phase, bytes):
        return phase.decode()
    return str(phase)


def evaluate_latent_sequence_model(
    checkpoint_path: str,
    source_path: str,
    output_path: str | None = None,
    *,
    device: str | None = None,
    small_thresholds=(1e-15, 1e-12),
    mech_path: str | None = None,
) -> dict:
    raw_sequences, times, species_names = load_sequence_arrays(source_path, dtype=np.float64)
    sequence_phase_name = _read_sequence_phase_name(source_path)
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "ae_latent_gru")

    if model_type in {"conserved_latent_delta", "stoichiometric_flux"}:
        state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
        state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
        log_dt_mean = np.asarray(checkpoint["log_dt_mean"], dtype=np.float64)
        log_dt_std = np.asarray(checkpoint["log_dt_std"], dtype=np.float64)
        state_std = np.where(state_std > 0, state_std, 1.0)
        log_dt_std = np.where(log_dt_std > 0, log_dt_std, 1.0)
        if model_type == "conserved_latent_delta":
            model = _load_conserved_model(checkpoint, raw_sequences.shape[-1], torch_device)
        else:
            model = _load_stoich_model(checkpoint, raw_sequences.shape[-1], torch_device)

        current = raw_sequences[:, :-1, :]
        log_dt_true = np.log(np.maximum(np.diff(times, axis=1), 1e-300))[..., None]
        current_norm = ((current - state_mean) / state_std).astype(np.float32)
        log_dt_norm = ((log_dt_true - log_dt_mean) / log_dt_std).astype(np.float32)
        current_tensor = torch.tensor(current_norm.reshape(-1, current_norm.shape[-1]), dtype=torch.float32, device=torch_device)
        current_y_tensor = torch.tensor(current[:, :, 2:].reshape(-1, current.shape[-1] - 2), dtype=torch.float64, device=torch_device)
        log_dt_tensor = torch.tensor(log_dt_norm.reshape(-1, 1), dtype=torch.float32, device=torch_device)

        with torch.no_grad():
            output = model(current_tensor, log_dt_tensor, current_species=current_y_tensor)
        pred_norm = output["next_state"].detach().cpu().numpy().reshape(current.shape)
        pred = np.empty_like(current, dtype=np.float64)
        pred[..., :2] = pred_norm[..., :2] * state_std[:2] + state_mean[:2]
        pred[..., 2:] = pred_norm[..., 2:]
        pred_log_dt = (output["log_dt"].detach().cpu().numpy().reshape(-1) * log_dt_std[0]) + log_dt_mean[0]
    else:
        model, checkpoint = _load_model(checkpoint_path, raw_sequences.shape[-1], torch_device)
        mean, std = _normalization_from_checkpoint(checkpoint, raw_sequences.shape[-1])

        normalized = (raw_sequences - mean) / std
        tensor = torch.tensor(normalized, dtype=torch.float32, device=torch_device)

        with torch.no_grad():
            pred_norm, z_next = model(tensor)
            time_event = model.predict_time_event(z_next)

        pred = pred_norm.detach().cpu().numpy() * std + mean
        pred_log_dt = time_event["log_dt"].detach().cpu().numpy().reshape(-1)

    true_next = raw_sequences[:, 1:, :]
    pred_next = pred

    true_species = true_next[:, :, 2:]
    pred_species = pred_next[:, :, 2:]
    true_state_flat = true_next.reshape(-1, true_next.shape[-1])
    pred_state_flat = pred_next.reshape(-1, pred_next.shape[-1])
    true_species_flat = true_species.reshape(-1, true_species.shape[-1])
    pred_species_flat = pred_species.reshape(-1, pred_species.shape[-1])

    true_dt = np.diff(times, axis=1).reshape(-1)
    pred_dt = np.exp(pred_log_dt)
    pred_dt_by_trajectory = pred_dt.reshape(raw_sequences.shape[0], raw_sequences.shape[1] - 1)
    true_horizon = times[:, -1] - times[:, 0]
    pred_horizon = np.sum(pred_dt_by_trajectory, axis=1)

    results = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "n_trajectories": int(raw_sequences.shape[0]),
        "n_steps": int(raw_sequences.shape[1] - 1),
        "latent_step_metrics": {
            "mode": "teacher_forced_saved_intervals",
            "steps_per_trajectory": int(raw_sequences.shape[1] - 1),
            "total_latent_transitions": int(raw_sequences.shape[0] * (raw_sequences.shape[1] - 1)),
            "mean_true_horizon": float(np.mean(true_horizon)),
            "mean_predicted_horizon": float(np.mean(pred_horizon)),
            "median_predicted_horizon": float(np.median(pred_horizon)),
            "note": "Current baseline uses one latent transition per saved time interval; adaptive latent step count is not learned yet.",
        },
        "input_dim": int(raw_sequences.shape[-1]),
        "species_names": species_names,
        "state_metrics": summarize_predictions(
            true_state_flat,
            pred_state_flat,
            species_names=["T", "P", *species_names],
            small_thresholds=small_thresholds,
        ),
        "species_metrics": summarize_predictions(
            true_species_flat,
            pred_species_flat,
            species_names=species_names,
            small_thresholds=small_thresholds,
        ),
        "time_metrics": _relative_time_error(true_dt, pred_dt),
        "checkpoint_training_config": checkpoint.get("training_config", {}),
        "checkpoint_final_metrics": checkpoint.get("final_metrics", {}),
    }

    if mech_path is not None:
        import cantera as ct

        phase_name = checkpoint.get("phase_name") or sequence_phase_name
        gas = ct.Solution(mech_path, phase_name) if phase_name is not None else ct.Solution(mech_path)
        results["conservation_metrics"] = conservation_summary(gas, true_species_flat, pred_species_flat)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_to_jsonable(results), indent=2, sort_keys=True))

    return results
