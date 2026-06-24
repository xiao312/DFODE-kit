from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dfode_kit.models.latent_baseline import StoichiometricFluxModel
from dfode_kit.physics.atom_conservation import stoichiometric_mass_fraction_matrix
from dfode_kit.training.conserved_sequence import _flatten_pairs
from dfode_kit.training.latent_sequence import load_sequence_arrays


@dataclass(frozen=True)
class StoichSequenceTrainingConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    time_weight: float = 0.01


def train_stoich_sequence_model(
    source_path: str,
    output_path: str,
    mech_path: str,
    *,
    config: StoichSequenceTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    import cantera as ct

    cfg = config or StoichSequenceTrainingConfig()
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
    log_dt_norm = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)

    gas = ct.Solution(mech_path)
    stoich_mass = stoichiometric_mass_fraction_matrix(gas)
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    tensors = torch.utils.data.TensorDataset(
        torch.tensor(current_norm, dtype=torch.float32),
        torch.tensor(target_norm, dtype=torch.float32),
        torch.tensor(current[:, 2:], dtype=torch.float64),
        torch.tensor(target[:, 2:], dtype=torch.float64),
        torch.tensor(log_dt_norm, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(tensors, batch_size=cfg.batch_size, shuffle=True)
    model = StoichiometricFluxModel(
        input_dim=raw_sequences.shape[-1],
        stoichiometric_mass_matrix=stoich_mass,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = float("nan")
    last_state_loss = float("nan")
    last_time_loss = float("nan")

    for _epoch in range(cfg.epochs):
        for current_batch, target_batch, current_y_batch, target_y_batch, log_dt_batch in loader:
            current_batch = current_batch.to(torch_device)
            target_batch = target_batch.to(torch_device)
            current_y_batch = current_y_batch.to(torch_device)
            target_y_batch = target_y_batch.to(torch_device)
            log_dt_batch = log_dt_batch.to(torch_device)
            output = model(current_batch, log_dt_batch, current_species=current_y_batch)
            tp_loss = torch.nn.functional.mse_loss(output["next_state"][..., :2], target_batch[..., :2].to(torch.float64))
            species_loss = torch.nn.functional.mse_loss(output["next_state"][..., 2:], target_y_batch)
            state_loss = tp_loss + species_loss
            time_loss = torch.nn.functional.mse_loss(output["log_dt"], log_dt_batch)
            loss = state_loss + cfg.time_weight * time_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            last_state_loss = float(state_loss.detach().cpu())
            last_time_loss = float(time_loss.detach().cpu())

    torch.save(
        {
            "model_type": "stoichiometric_flux",
            "net": model.state_dict(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "log_dt_mean": log_dt_mean.tolist(),
            "log_dt_std": log_dt_std.tolist(),
            "species_names": species_names,
            "mechanism": mech_path,
            "stoichiometric_mass_matrix": stoich_mass.tolist(),
            "training_config": {
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "time_weight": cfg.time_weight,
            },
            "final_metrics": {
                "loss": last_loss,
                "state_loss": last_state_loss,
                "time_loss": last_time_loss,
            },
        },
        output_path,
    )

    return {
        "loss": last_loss,
        "state_loss": last_state_loss,
        "time_loss": last_time_loss,
    }
