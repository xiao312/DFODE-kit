from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from dfode_kit.data.interval_pairs import load_interval_pair_arrays, load_interval_thermo_arrays
from dfode_kit.models.positive_interval import (
    NeuralPatankarIntervalModel,
    ReactionTrajectoryFreeEnergyModel,
)
from dfode_kit.physics.atom_conservation import reaction_affinity_over_rt, reaction_stoichiometry


@dataclass
class PositiveIntervalTrainingConfig:
    variant: str = "neural-patankar"
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    latent_dim: int = 32
    transform_alpha: float = 0.1
    delta_loss_weight: float = 0.1
    extent_scale: float = 1e-4
    trajectory_scale: float = 1e-5
    mobility_scale: float = 1e-4
    proximal_steps: int = 3
    proximal_beta: float = 1.0
    positivity_floor: float = 1e-30
    seed: int = 260624
    deterministic: bool = True
    log_every: int = 10


def _set_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False


def _signed_power(values: torch.Tensor, alpha: float) -> torch.Tensor:
    magnitude = torch.abs(values)
    safe_magnitude = magnitude.clamp_min(torch.finfo(values.dtype).tiny)
    transformed = torch.sign(values) * torch.pow(safe_magnitude, alpha) / alpha
    return torch.where(magnitude == 0.0, torch.zeros_like(transformed), transformed)


def _affinity_features(source: str, current: np.ndarray, gas) -> np.ndarray:
    try:
        affinity, _activity, _reversible = load_interval_thermo_arrays(source, dtype=np.float32)
        return affinity
    except (KeyError, OSError, ValueError):
        stoich = reaction_stoichiometry(gas)
        affinity = np.empty((current.shape[0], gas.n_reactions), dtype=np.float32)
        for index, state in enumerate(current):
            gas.TPY = float(state[0]), float(state[1]), state[2:]
            mu_over_rt = gas.chemical_potentials / (gas_constant() * gas.T)
            affinity[index] = reaction_affinity_over_rt(stoich.net, mu_over_rt)
        return affinity


def gas_constant() -> float:
    import cantera as ct

    return float(ct.gas_constant)


def _build_model(config, gas, input_dim):
    stoich = reaction_stoichiometry(gas)
    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    if config.variant == "neural-patankar":
        return NeuralPatankarIntervalModel(
            input_dim=input_dim,
            reactant_mass_matrix=molecular_weights[:, None] * stoich.reactants,
            product_mass_matrix=molecular_weights[:, None] * stoich.products,
            reaction_reversible=stoich.reversible,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            extent_scale=config.extent_scale,
            availability_floor=config.positivity_floor,
        )
    if config.variant == "reaction-trajectory":
        return ReactionTrajectoryFreeEnergyModel(
            input_dim=input_dim,
            stoichiometric_matrix=stoich.net,
            molecular_weights=molecular_weights,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            trajectory_scale=config.trajectory_scale,
            mobility_scale=config.mobility_scale,
            proximal_steps=config.proximal_steps,
            proximal_beta=config.proximal_beta,
            positivity_floor=config.positivity_floor,
        )
    raise ValueError(f"unsupported positive interval variant: {config.variant}")


def train_positive_interval_model(
    train_source: str,
    validation_source: str,
    mech_path: str,
    output_path: str,
    *,
    phase_name: str | None = None,
    device: str | None = None,
    config: PositiveIntervalTrainingConfig | None = None,
) -> dict:
    import cantera as ct

    config = config or PositiveIntervalTrainingConfig()
    _set_seed(config.seed, config.deterministic)
    train_current, train_target, train_dt, *_rest = load_interval_pair_arrays(train_source, dtype=np.float64)
    val_current, val_target, val_dt, *_val_rest = load_interval_pair_arrays(validation_source, dtype=np.float64)
    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    if train_current.shape[1] != gas.n_species + 2:
        raise ValueError("dataset species dimension does not match mechanism")

    state_mean = np.mean(train_current, axis=0)
    state_std = np.std(train_current, axis=0)
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    train_log_dt = np.log(np.maximum(train_dt, 1e-300))[:, None]
    val_log_dt = np.log(np.maximum(val_dt, 1e-300))[:, None]
    log_dt_mean = np.mean(train_log_dt, axis=0)
    log_dt_std = np.std(train_log_dt, axis=0)
    log_dt_std = np.where(log_dt_std > 0.0, log_dt_std, 1.0)

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, gas, train_current.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_x = torch.tensor((train_current - state_mean) / state_std, dtype=torch.float32, device=torch_device)
    train_t = torch.tensor((train_log_dt - log_dt_mean) / log_dt_std, dtype=torch.float32, device=torch_device)
    train_y0 = torch.tensor(train_current[:, 2:], dtype=torch.float64, device=torch_device)
    train_true = torch.tensor(train_target, dtype=torch.float64, device=torch_device)
    val_x = torch.tensor((val_current - state_mean) / state_std, dtype=torch.float32, device=torch_device)
    val_t = torch.tensor((val_log_dt - log_dt_mean) / log_dt_std, dtype=torch.float32, device=torch_device)
    val_y0 = torch.tensor(val_current[:, 2:], dtype=torch.float64, device=torch_device)
    val_true = torch.tensor(val_target, dtype=torch.float64, device=torch_device)

    train_affinity = val_affinity = None
    if config.variant == "reaction-trajectory":
        train_affinity = torch.tensor(
            _affinity_features(train_source, train_current, gas),
            dtype=torch.float32,
            device=torch_device,
        )
        val_affinity = torch.tensor(
            _affinity_features(validation_source, val_current, gas),
            dtype=torch.float32,
            device=torch_device,
        )

    state_mean_t = torch.tensor(state_mean, dtype=torch.float64, device=torch_device)
    state_std_t = torch.tensor(state_std, dtype=torch.float64, device=torch_device)

    def loss_for(indices, training):
        x = train_x[indices] if training else val_x[indices]
        t = train_t[indices] if training else val_t[indices]
        y0 = train_y0[indices] if training else val_y0[indices]
        truth = train_true[indices] if training else val_true[indices]
        if config.variant == "reaction-trajectory":
            affinity = train_affinity[indices] if training else val_affinity[indices]
            output = model(x, t, affinity, current_species=y0)
        else:
            output = model(x, t, current_species=y0)
        prediction = output["next_state"]
        true_tp_norm = (truth[:, :2] - state_mean_t[:2]) / state_std_t[:2]
        tp_loss = F.l1_loss(prediction[:, :2], true_tp_norm)
        pred_y = prediction[:, 2:]
        true_y = truth[:, 2:]
        y_loss = F.l1_loss(
            _signed_power(pred_y, config.transform_alpha),
            _signed_power(true_y, config.transform_alpha),
        )
        delta_loss = F.l1_loss(
            _signed_power(pred_y - y0, config.transform_alpha),
            _signed_power(true_y - y0, config.transform_alpha),
        )
        return tp_loss + y_loss + config.delta_loss_weight * delta_loss, output

    history = []
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator, device=torch_device)
        total = 0.0
        for start in range(0, order.numel(), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _output = loss_for(indices, True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * indices.numel()
        train_loss = total / order.numel()
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                negative = 0
                count = 0
                free_energy_positive = 0
                for start in range(0, val_x.shape[0], config.batch_size):
                    indices = torch.arange(
                        start,
                        min(start + config.batch_size, val_x.shape[0]),
                        device=torch_device,
                    )
                    val_loss, output = loss_for(indices, False)
                    val_total += float(val_loss) * indices.numel()
                    negative += int(torch.sum(output["next_species"] < 0.0))
                    count += output["next_species"].numel()
                    if "local_free_energy_delta" in output:
                        free_energy_positive += int(torch.sum(output["local_free_energy_delta"] > 1e-12))
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": val_total / val_x.shape[0],
                    "negative_entry_rate": negative / count,
                    "positive_free_energy_count": free_energy_positive,
                }
                history.append(row)
                print(json.dumps(row), flush=True)

    checkpoint = {
        "positive_model_type": config.variant,
        "net": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "log_dt_mean": log_dt_mean,
        "log_dt_std": log_dt_std,
        "species_names": list(gas.species_names),
        "phase_name": gas.name,
        "stoichiometric_matrix": reaction_stoichiometry(gas).net,
        "molecular_weights": np.asarray(gas.molecular_weights, dtype=np.float64),
        "reaction_reversible": reaction_stoichiometry(gas).reversible,
        "training_config": asdict(config),
        "history": history,
        "train_source": train_source,
        "validation_source": validation_source,
        "mechanism": mech_path,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return history[-1]
