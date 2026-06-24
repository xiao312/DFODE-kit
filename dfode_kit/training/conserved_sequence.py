from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dfode_kit.models.latent_baseline import ConservedLatentDeltaModel, signed_power_transform
from dfode_kit.physics.atom_conservation import completion_matrix_for_mass_fractions
from dfode_kit.training.latent_sequence import load_sequence_arrays


@dataclass(frozen=True)
class ConservedSequenceTrainingConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    time_weight: float = 0.01
    key_mode: str = "signed-power"
    transform_alpha: float = 0.1
    transform_eps: float = 1e-12
    key_scale_alpha: float = 0.0
    key_loss_weight: float = 0.1
    species_loss_mode: str = "signed-power-plus-delta"
    species_transform_alpha: float = 0.1
    species_loss_weight: float = 0.1
    loss_kind: str = "mae"
    transform_scale_by_alpha: bool = True


def _signed_power_transform_np(
    value: np.ndarray,
    *,
    alpha: float,
    eps: float,
    scale_by_alpha: bool = False,
) -> np.ndarray:
    transformed = np.sign(value) * (np.abs(value) ** alpha)
    if scale_by_alpha:
        transformed = transformed / alpha
    return transformed


def _regression_loss(pred, target, *, loss_kind: str):
    if loss_kind == "mse":
        return torch.nn.functional.mse_loss(pred, target)
    if loss_kind == "mae":
        return torch.nn.functional.l1_loss(pred, target)
    raise ValueError(f"Unsupported loss_kind: {loss_kind}")


def _flatten_pairs(raw_sequences: np.ndarray, times: np.ndarray):
    current = raw_sequences[:, :-1, :]
    target = raw_sequences[:, 1:, :]
    log_dt = np.log(np.maximum(np.diff(times, axis=1), 1e-300))[..., None]
    return (
        current.reshape(-1, current.shape[-1]).astype(np.float64),
        target.reshape(-1, target.shape[-1]).astype(np.float64),
        log_dt.reshape(-1, 1).astype(np.float64),
    )


def train_conserved_sequence_model(
    source_path: str,
    output_path: str,
    mech_path: str,
    *,
    config: ConservedSequenceTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    import cantera as ct

    cfg = config or ConservedSequenceTrainingConfig()
    raw_sequences, times, species_names = load_sequence_arrays(source_path, dtype=np.float64)
    current, target, log_dt = _flatten_pairs(raw_sequences, times)

    state_mean = current.mean(axis=0)
    state_std = current.std(axis=0)
    state_std = np.where(state_std > 0, state_std, 1.0)
    log_dt_mean = log_dt.mean(axis=0)
    log_dt_std = log_dt.std(axis=0)
    log_dt_std = np.where(log_dt_std > 0, log_dt_std, 1.0)

    current_norm = ((current - state_mean) / state_std).astype(np.float32)
    target_norm = ((target - state_mean) / state_std).astype(np.float32)
    log_dt_norm = (log_dt - log_dt_mean) / log_dt_std

    gas = ct.Solution(mech_path)
    completion = completion_matrix_for_mass_fractions(gas)
    key_species_indices = np.asarray(completion.key_species_indices, dtype=np.int64)
    target_key_delta = target[:, 2 + key_species_indices] - current[:, 2 + key_species_indices]
    if cfg.key_mode == "signed-power":
        key_target = _signed_power_transform_np(
            target_key_delta,
            alpha=cfg.transform_alpha,
            eps=cfg.transform_eps,
            scale_by_alpha=cfg.transform_scale_by_alpha,
        )
    else:
        key_target = target_key_delta
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    tensors = torch.utils.data.TensorDataset(
        torch.tensor(current_norm, dtype=torch.float32),
        torch.tensor(target_norm, dtype=torch.float32),
        torch.tensor(current[:, 2:], dtype=torch.float64),
        torch.tensor(target[:, 2:], dtype=torch.float64),
        torch.tensor(key_target, dtype=torch.float32),
        torch.tensor(log_dt_norm, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(tensors, batch_size=cfg.batch_size, shuffle=True)
    model = ConservedLatentDeltaModel(
        input_dim=raw_sequences.shape[-1],
        completion_matrix=completion.matrix,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        key_mode=cfg.key_mode,
        transform_alpha=cfg.transform_alpha,
        transform_eps=cfg.transform_eps,
        transform_scale_by_alpha=cfg.transform_scale_by_alpha,
        key_species_indices=completion.key_species_indices,
        key_scale_alpha=cfg.key_scale_alpha,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = float("nan")
    last_state_loss = float("nan")
    last_time_loss = float("nan")
    last_key_loss = float("nan")

    for _epoch in range(cfg.epochs):
        for current_batch, target_batch, current_y_batch, target_y_batch, key_target_batch, log_dt_batch in loader:
            current_batch = current_batch.to(torch_device)
            target_batch = target_batch.to(torch_device)
            current_y_batch = current_y_batch.to(torch_device)
            target_y_batch = target_y_batch.to(torch_device)
            key_target_batch = key_target_batch.to(torch_device)
            log_dt_batch = log_dt_batch.to(torch_device)
            output = model(current_batch, log_dt_batch, current_species=current_y_batch)
            tp_loss = _regression_loss(
                output["next_state"][..., :2],
                target_batch[..., :2].to(torch.float64),
                loss_kind=cfg.loss_kind,
            )
            physical_species_loss = _regression_loss(
                output["next_state"][..., 2:],
                target_y_batch,
                loss_kind=cfg.loss_kind,
            )
            transformed_species_loss = _regression_loss(
                signed_power_transform(
                    output["next_state"][..., 2:],
                    alpha=cfg.species_transform_alpha,
                    eps=cfg.transform_eps,
                    scale_by_alpha=cfg.transform_scale_by_alpha,
                ),
                signed_power_transform(
                    target_y_batch,
                    alpha=cfg.species_transform_alpha,
                    eps=cfg.transform_eps,
                    scale_by_alpha=cfg.transform_scale_by_alpha,
                ),
                loss_kind=cfg.loss_kind,
            )
            transformed_species_delta_loss = _regression_loss(
                signed_power_transform(
                    output["delta_y"],
                    alpha=cfg.species_transform_alpha,
                    eps=cfg.transform_eps,
                    scale_by_alpha=cfg.transform_scale_by_alpha,
                ),
                signed_power_transform(
                    target_y_batch - current_y_batch,
                    alpha=cfg.species_transform_alpha,
                    eps=cfg.transform_eps,
                    scale_by_alpha=cfg.transform_scale_by_alpha,
                ),
                loss_kind=cfg.loss_kind,
            )
            if cfg.species_loss_mode == "physical":
                species_loss = physical_species_loss
            elif cfg.species_loss_mode == "signed-power":
                species_loss = transformed_species_loss
            elif cfg.species_loss_mode == "signed-power-delta":
                species_loss = transformed_species_delta_loss
            elif cfg.species_loss_mode == "physical-plus-signed-power":
                species_loss = physical_species_loss + cfg.species_loss_weight * transformed_species_loss
            elif cfg.species_loss_mode == "signed-power-plus-delta":
                species_loss = transformed_species_loss + cfg.species_loss_weight * transformed_species_delta_loss
            else:
                raise ValueError(f"Unsupported species_loss_mode: {cfg.species_loss_mode}")
            state_loss = tp_loss + species_loss
            time_loss = _regression_loss(output["log_dt"], log_dt_batch, loss_kind=cfg.loss_kind)
            if cfg.key_mode == "signed-power":
                key_prediction = output["raw_key"]
            else:
                key_prediction = output["delta_key"]
            key_loss = _regression_loss(key_prediction, key_target_batch, loss_kind=cfg.loss_kind)
            loss = state_loss + cfg.time_weight * time_loss + cfg.key_loss_weight * key_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            last_state_loss = float(state_loss.detach().cpu())
            last_time_loss = float(time_loss.detach().cpu())
            last_key_loss = float(key_loss.detach().cpu())

    torch.save(
        {
            "model_type": "conserved_latent_delta",
            "net": model.state_dict(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "log_dt_mean": log_dt_mean.tolist(),
            "log_dt_std": log_dt_std.tolist(),
            "species_names": species_names,
            "mechanism": mech_path,
            "completion_matrix": completion.matrix.tolist(),
            "completion_basis": "mass_fraction_element_matrix",
            "key_species_indices": list(completion.key_species_indices),
            "dependent_species_indices": list(completion.dependent_species_indices),
            "training_config": {
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "time_weight": cfg.time_weight,
                "key_mode": cfg.key_mode,
                "transform_alpha": cfg.transform_alpha,
                "transform_eps": cfg.transform_eps,
                "key_scale_alpha": cfg.key_scale_alpha,
                "key_loss_weight": cfg.key_loss_weight,
                "species_loss_mode": cfg.species_loss_mode,
                "species_transform_alpha": cfg.species_transform_alpha,
                "species_loss_weight": cfg.species_loss_weight,
                "loss_kind": cfg.loss_kind,
                "transform_scale_by_alpha": cfg.transform_scale_by_alpha,
            },
            "final_metrics": {
                "loss": last_loss,
                "state_loss": last_state_loss,
                "time_loss": last_time_loss,
                "key_loss": last_key_loss,
            },
        },
        output_path,
    )

    return {
        "loss": last_loss,
        "state_loss": last_state_loss,
        "time_loss": last_time_loss,
        "key_loss": last_key_loss,
    }
