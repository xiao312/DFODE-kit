from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from dfode_kit.evaluation.conservation import conservation_summary
from dfode_kit.evaluation.metrics import summarize_predictions
from dfode_kit.models.latent_baseline import signed_power_transform
from dfode_kit.models.unified_conserved import SharedLatentDynamicsConservedModel
from dfode_kit.physics.atom_conservation import completion_matrix_for_mass_fractions
from dfode_kit.training.conserved_sequence import _regression_loss, _signed_power_transform_np
from dfode_kit.training.latent_sequence import load_sequence_arrays


@dataclass(frozen=True)
class UnifiedConservedTrainingConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    time_weight: float = 0.01
    transform_alpha: float = 0.1
    key_loss_weight: float = 0.1
    species_loss_weight: float = 0.1
    loss_kind: str = "mae"
    transform_scale_by_alpha: bool = True


def _mechanism_group(case: dict) -> str:
    phase = case.get("phase_name")
    if phase:
        return f"{case['mechanism']}::{phase}"
    return case["mechanism"]


def _solution_for_case(case: dict):
    import cantera as ct

    phase = case.get("phase_name")
    if phase:
        return ct.Solution(case["mechanism"], phase)
    return ct.Solution(case["mechanism"])


def _flatten_pairs(raw_sequences: np.ndarray, times: np.ndarray):
    current = raw_sequences[:, :-1, :]
    target = raw_sequences[:, 1:, :]
    log_dt = np.log(np.maximum(np.diff(times, axis=1), 1e-300))[..., None]
    return (
        current.reshape(-1, current.shape[-1]).astype(np.float64),
        target.reshape(-1, target.shape[-1]).astype(np.float64),
        log_dt.reshape(-1, 1).astype(np.float64),
    )


def _load_group_arrays(cases: list[dict], split: str):
    currents = []
    targets = []
    log_dts = []
    species_names = None
    for case in cases:
        path = case["splits"][split]["path"]
        raw, times, names = load_sequence_arrays(path, dtype=np.float64)
        if species_names is None:
            species_names = names
        elif species_names != names:
            raise ValueError(f"Species names mismatch within mechanism group {_mechanism_group(case)}")
        current, target, log_dt = _flatten_pairs(raw, times)
        currents.append(current)
        targets.append(target)
        log_dts.append(log_dt)
    return np.concatenate(currents), np.concatenate(targets), np.concatenate(log_dts), species_names


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def train_unified_conserved_sequence_model(
    manifest_path: str,
    output_path: str,
    eval_output_path: str | None = None,
    *,
    config: UnifiedConservedTrainingConfig | None = None,
    device: str | None = None,
) -> dict:
    cfg = config or UnifiedConservedTrainingConfig()
    manifest = json.loads(Path(manifest_path).read_text())
    cases = manifest["cases"]
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(_mechanism_group(case), []).append(case)

    group_data = {}
    mechanism_configs = {}
    loaders = {}
    for name, group_cases in groups.items():
        current, target, log_dt, species_names = _load_group_arrays(group_cases, "train")
        state_mean = current.mean(axis=0)
        state_std = np.where(current.std(axis=0) > 0, current.std(axis=0), 1.0)
        log_dt_mean = log_dt.mean(axis=0)
        log_dt_std = np.where(log_dt.std(axis=0) > 0, log_dt.std(axis=0), 1.0)
        gas = _solution_for_case(group_cases[0])
        completion = completion_matrix_for_mass_fractions(gas)
        key_indices = np.asarray(completion.key_species_indices, dtype=np.int64)
        key_delta = target[:, 2 + key_indices] - current[:, 2 + key_indices]
        key_target = _signed_power_transform_np(
            key_delta,
            alpha=cfg.transform_alpha,
            eps=0.0,
            scale_by_alpha=cfg.transform_scale_by_alpha,
        )
        tensors = torch.utils.data.TensorDataset(
            torch.tensor(((current - state_mean) / state_std).astype(np.float32), dtype=torch.float32),
            torch.tensor(((target - state_mean) / state_std).astype(np.float32), dtype=torch.float32),
            torch.tensor(current[:, 2:], dtype=torch.float64),
            torch.tensor(target[:, 2:], dtype=torch.float64),
            torch.tensor(key_target, dtype=torch.float32),
            torch.tensor(((log_dt - log_dt_mean) / log_dt_std).astype(np.float32), dtype=torch.float32),
        )
        loaders[name] = torch.utils.data.DataLoader(tensors, batch_size=cfg.batch_size, shuffle=True)
        group_data[name] = {
            "cases": group_cases,
            "species_names": species_names,
            "state_mean": state_mean,
            "state_std": state_std,
            "log_dt_mean": log_dt_mean,
            "log_dt_std": log_dt_std,
            "completion_matrix": completion.matrix,
            "key_species_indices": list(completion.key_species_indices),
            "dependent_species_indices": list(completion.dependent_species_indices),
        }
        mechanism_configs[name] = {
            "input_dim": current.shape[-1],
            "completion_matrix": completion.matrix,
        }

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = SharedLatentDynamicsConservedModel(
        mechanism_configs,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        transform_alpha=cfg.transform_alpha,
        transform_scale_by_alpha=cfg.transform_scale_by_alpha,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = float("nan")
    for _epoch in range(cfg.epochs):
        for name, loader in loaders.items():
            for current_batch, target_batch, current_y_batch, target_y_batch, key_target_batch, log_dt_batch in loader:
                current_batch = current_batch.to(torch_device)
                target_batch = target_batch.to(torch_device)
                current_y_batch = current_y_batch.to(torch_device)
                target_y_batch = target_y_batch.to(torch_device)
                key_target_batch = key_target_batch.to(torch_device)
                log_dt_batch = log_dt_batch.to(torch_device)
                output = model(name, current_batch, log_dt_batch, current_species=current_y_batch)
                tp_loss = _regression_loss(
                    output["next_state"][..., :2],
                    target_batch[..., :2].to(torch.float64),
                    loss_kind=cfg.loss_kind,
                )
                y_loss = _regression_loss(
                    signed_power_transform(
                        output["next_state"][..., 2:],
                        alpha=cfg.transform_alpha,
                        scale_by_alpha=cfg.transform_scale_by_alpha,
                    ),
                    signed_power_transform(
                        target_y_batch,
                        alpha=cfg.transform_alpha,
                        scale_by_alpha=cfg.transform_scale_by_alpha,
                    ),
                    loss_kind=cfg.loss_kind,
                )
                delta_loss = _regression_loss(
                    signed_power_transform(
                        output["delta_y"],
                        alpha=cfg.transform_alpha,
                        scale_by_alpha=cfg.transform_scale_by_alpha,
                    ),
                    signed_power_transform(
                        target_y_batch - current_y_batch,
                        alpha=cfg.transform_alpha,
                        scale_by_alpha=cfg.transform_scale_by_alpha,
                    ),
                    loss_kind=cfg.loss_kind,
                )
                key_loss = _regression_loss(output["raw_key"], key_target_batch, loss_kind=cfg.loss_kind)
                time_loss = _regression_loss(output["log_dt"], log_dt_batch, loss_kind=cfg.loss_kind)
                loss = tp_loss + y_loss + cfg.species_loss_weight * delta_loss + cfg.key_loss_weight * key_loss + cfg.time_weight * time_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())

    checkpoint = {
        "model_type": "shared_latent_dynamics_conserved",
        "net": model.state_dict(),
        "manifest": manifest_path,
        "groups": {
            name: {
                "cases": [case["case_id"] for case in data["cases"]],
                "state_mean": data["state_mean"].tolist(),
                "state_std": data["state_std"].tolist(),
                "log_dt_mean": data["log_dt_mean"].tolist(),
                "log_dt_std": data["log_dt_std"].tolist(),
                "species_names": data["species_names"],
                "completion_matrix": data["completion_matrix"].tolist(),
                "key_species_indices": data["key_species_indices"],
                "dependent_species_indices": data["dependent_species_indices"],
            }
            for name, data in group_data.items()
        },
        "training_config": cfg.__dict__,
        "final_metrics": {"loss": last_loss},
    }
    torch.save(checkpoint, output_path)

    results = {
        "checkpoint": output_path,
        "manifest": manifest_path,
        "training_config": cfg.__dict__,
        "final_metrics": {"loss": last_loss},
        "groups": {},
    }
    for name, data in group_data.items():
        current, target, log_dt, species_names = _load_group_arrays(data["cases"], "val")
        state_mean = data["state_mean"]
        state_std = data["state_std"]
        log_dt_mean = data["log_dt_mean"]
        log_dt_std = data["log_dt_std"]
        current_norm = ((current - state_mean) / state_std).astype(np.float32)
        log_dt_norm = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)
        with torch.no_grad():
            output = model(
                name,
                torch.tensor(current_norm, dtype=torch.float32, device=torch_device),
                torch.tensor(log_dt_norm, dtype=torch.float32, device=torch_device),
                current_species=torch.tensor(current[:, 2:], dtype=torch.float64, device=torch_device),
            )
        pred = output["next_state"].detach().cpu().numpy()
        true_species = target[:, 2:]
        pred_species = pred[:, 2:]
        gas = _solution_for_case(data["cases"][0])
        pred_log_dt = output["log_dt"].detach().cpu().numpy().reshape(-1) * log_dt_std[0] + log_dt_mean[0]
        true_dt = np.exp(log_dt.reshape(-1))
        pred_dt = np.exp(pred_log_dt)
        rel_dt = np.abs(pred_dt - true_dt) / np.maximum(np.abs(true_dt), 1e-300)
        results["groups"][name] = {
            "cases": [case["case_id"] for case in data["cases"]],
            "n_val_pairs": int(current.shape[0]),
            "species_metrics": summarize_predictions(true_species, pred_species, species_names=species_names),
            "conservation_metrics": conservation_summary(gas, true_species, pred_species),
            "time_metrics": {
                "mae_log_dt": float(np.mean(np.abs(np.log(np.maximum(pred_dt, 1e-300)) - np.log(np.maximum(true_dt, 1e-300))))),
                "mae_dt": float(np.mean(np.abs(pred_dt - true_dt))),
                "median_relative_dt_error": float(np.median(rel_dt)),
                "mean_relative_dt_error": float(np.mean(rel_dt)),
            },
        }

    if eval_output_path is not None:
        Path(eval_output_path).write_text(json.dumps(_jsonable(results), indent=2, sort_keys=True))
    return results
