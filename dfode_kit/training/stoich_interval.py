from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from dfode_kit.data.interval_pairs import (
    load_interval_midpoint_arrays,
    load_interval_pair_arrays,
    load_interval_thermo_arrays,
)
from dfode_kit.models.latent_baseline import (
    LatentSubstepStoichiometricIntervalModel,
    StoichiometricIntervalModel,
    ThermoStoichiometricIntervalModel,
    signed_power_transform,
)
try:
    from dfode_kit.models.latent_baseline import ThermoProgressSubstepStoichiometricIntervalModel
except ImportError:  # pragma: no cover - allows integration to land before model-owner code.
    ThermoProgressSubstepStoichiometricIntervalModel = None
from dfode_kit.physics.atom_conservation import reaction_stoichiometry, stoichiometric_mass_fraction_matrix
from dfode_kit.training.conserved_sequence import _regression_loss


THERMO_FEATURE_VARIANTS = {"thermo-affinity", "substep-soft-thermo", "thermo-progress-substep"}
SUBSTEP_VARIANTS = {"substep", "substep-soft-thermo", "thermo-progress-substep"}


@dataclass(frozen=True)
class StoichIntervalTrainingConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    epochs: int = 100
    batch_size: int = 4096
    lr: float = 1e-3
    transform_alpha: float = 0.1
    species_loss_weight: float = 0.1
    loss_kind: str = "mae"
    transform_scale_by_alpha: bool = True
    flux_mode: str = "signed-power"
    input_perturb_alpha: float = 0.0
    input_perturb_seed: int = 20260626
    model_variant: str = "stoich"
    thermo_extent_scale: float = 1e-3
    thermo_force_scale: float = 1.0
    thermo_availability_threshold: float = -60.0
    thermo_availability_slope: float = 1.0
    substeps: int = 2
    adaptive_substeps: bool = False
    semigroup_loss_weight: float = 0.0
    midpoint_loss_weight: float = 0.0
    thermo_embedding_dim: int = 32
    log_every_epochs: int = 50
    gradient_clip_norm: float = 0.0
    reaction_extent_raw_clip: float = 0.0
    thermo_feature_clip: float = 10.0
    skip_nonfinite_batches: bool = True
    error_loss_weight: float = 0.0
    step_policy_loss_weight: float = 0.0
    learned_adaptive_substeps: bool = False
    thermo_progress_force_clip: float = 8.0
    thermo_progress_mobility_scale: float = 1.0
    thermo_progress_mode: str = "sinh"
    latent_update_mode: str = "residual"
    latent_step_scale: float = 1.0
    proximal_correction_steps: int = 0
    proximal_correction_lr: float = 0.1
    proximal_free_energy_weight: float = 0.0
    proximal_y_floor: float = 1e-300
    positivity_safety_factor: float = 0.0


def _substeps_for_dt(dt_values: torch.Tensor) -> torch.Tensor:
    result = torch.ones_like(dt_values, dtype=torch.long)
    result = torch.where(dt_values >= 1e-7, torch.full_like(result, 2), result)
    result = torch.where(dt_values >= 1e-5, torch.full_like(result, 4), result)
    result = torch.where(dt_values >= 1e-4, torch.full_like(result, 8), result)
    return result


def _substep_labels(substeps: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(substeps, dtype=torch.long)
    labels = torch.where(substeps == 2, torch.ones_like(labels), labels)
    labels = torch.where(substeps == 4, torch.full_like(labels, 2), labels)
    labels = torch.where(substeps == 8, torch.full_like(labels, 3), labels)
    return labels


def _physical_to_normalized_state(state: torch.Tensor, state_mean: torch.Tensor, state_std: torch.Tensor) -> torch.Tensor:
    tp = state[..., :2]
    y = state[..., 2:]
    return torch.cat(
        [
            tp.to(torch.float32),
            ((y - state_mean[..., 2:]) / state_std[..., 2:]).to(torch.float32),
        ],
        dim=-1,
    ).to(torch.float32)


def _normalized_log_dt(dt_values, log_dt_mean, log_dt_std) -> np.ndarray:
    log_dt = np.log(np.maximum(dt_values, 1e-300))[:, None]
    return ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)


def _forward_interval_model(
    model,
    current_batch,
    log_dt_batch,
    current_y_batch,
    *,
    model_variant: str,
    affinity_batch=None,
    log_activity_batch=None,
    log_dt_step_batch=None,
    substeps_batch=None,
):
    if model_variant == "thermo-affinity":
        return model(
            current_batch,
            log_dt_batch,
            affinity_batch,
            log_activity_batch,
            current_species=current_y_batch,
        )
    if model_variant == "thermo-progress-substep":
        if substeps_batch is None:
            return model(
                current_batch,
                log_dt_batch,
                affinity_batch,
                log_activity_batch,
                current_species=current_y_batch,
                log_dt_step=log_dt_step_batch,
            )
        outputs = {}
        next_state = torch.empty(
            (current_batch.shape[0], current_batch.shape[1]),
            dtype=torch.float64,
            device=current_batch.device,
        )
        delta_y = torch.empty(
            (current_batch.shape[0], current_y_batch.shape[1]),
            dtype=torch.float64,
            device=current_batch.device,
        )
        for k_value in torch.unique(substeps_batch).detach().cpu().tolist():
            mask = substeps_batch == int(k_value)
            out = model(
                current_batch[mask],
                log_dt_batch[mask],
                affinity_batch[mask],
                log_activity_batch[mask],
                current_species=current_y_batch[mask],
                log_dt_step=log_dt_step_batch[mask] if log_dt_step_batch is not None else None,
                substeps=int(k_value),
            )
            next_state[mask] = out["next_state"]
            delta_y[mask] = out["delta_y"]
        outputs["next_state"] = next_state
        outputs["delta_y"] = delta_y
        for optional_key in ("error_estimate", "step_policy_logits"):
            if optional_key in out:
                values = torch.empty(
                    (current_batch.shape[0], out[optional_key].shape[-1]),
                    dtype=out[optional_key].dtype,
                    device=current_batch.device,
                )
                for k_value in torch.unique(substeps_batch).detach().cpu().tolist():
                    mask = substeps_batch == int(k_value)
                    local_out = model(
                        current_batch[mask],
                        log_dt_batch[mask],
                        affinity_batch[mask],
                        log_activity_batch[mask],
                        current_species=current_y_batch[mask],
                        log_dt_step=log_dt_step_batch[mask] if log_dt_step_batch is not None else None,
                        substeps=int(k_value),
                    )
                    values[mask] = local_out[optional_key]
                outputs[optional_key] = values
        return outputs
    if model_variant == "substep-soft-thermo":
        thermo_features = torch.cat([affinity_batch, log_activity_batch], dim=-1)
    else:
        thermo_features = None
    if model_variant in {"substep", "substep-soft-thermo"}:
        if substeps_batch is None:
            return model(
                current_batch,
                log_dt_batch,
                current_species=current_y_batch,
                log_dt_step=log_dt_step_batch,
                thermo_features=thermo_features,
            )
        outputs = {}
        next_state = torch.empty(
            (current_batch.shape[0], current_batch.shape[1]),
            dtype=torch.float64,
            device=current_batch.device,
        )
        delta_y = torch.empty(
            (current_batch.shape[0], current_y_batch.shape[1]),
            dtype=torch.float64,
            device=current_batch.device,
        )
        for k_value in torch.unique(substeps_batch).detach().cpu().tolist():
            mask = substeps_batch == int(k_value)
            local_features = thermo_features[mask] if thermo_features is not None else None
            out = model(
                current_batch[mask],
                log_dt_batch[mask],
                current_species=current_y_batch[mask],
                log_dt_step=log_dt_step_batch[mask] if log_dt_step_batch is not None else None,
                substeps=int(k_value),
                thermo_features=local_features,
            )
            next_state[mask] = out["next_state"]
            delta_y[mask] = out["delta_y"]
        outputs["next_state"] = next_state
        outputs["delta_y"] = delta_y
        return outputs
    return model(current_batch, log_dt_batch, current_species=current_y_batch)


def _perturb_current_state(
    current_state: torch.Tensor,
    *,
    state_std: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if alpha <= 0.0:
        return current_state

    perturbed = current_state.clone()
    dtype = perturbed.dtype
    device = perturbed.device
    tp_noise = torch.randn(
        perturbed[..., :2].shape,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    perturbed[..., :2] = perturbed[..., :2] + alpha * state_std[..., :2].to(device=device, dtype=dtype) * tp_noise
    perturbed[..., 0] = torch.clamp(perturbed[..., 0], min=250.0)
    perturbed[..., 1] = torch.clamp(perturbed[..., 1], min=1.0)

    y = torch.clamp(perturbed[..., 2:], min=0.0)
    log_noise = torch.randn(y.shape, dtype=dtype, device=device, generator=generator)
    y = y * torch.exp(alpha * log_noise)
    y_sum = torch.sum(y, dim=-1, keepdim=True)
    y = torch.where(y_sum > 0.0, y / y_sum, torch.softmax(log_noise, dim=-1))
    perturbed[..., 2:] = y
    return perturbed


def _phase_from_attrs(attrs: dict) -> str | None:
    phase = attrs.get("phase_name")
    if phase is None:
        return None
    if isinstance(phase, bytes):
        return phase.decode("utf-8")
    text = str(phase)
    return text if text else None


def train_stoich_interval_model(
    source_path: str,
    output_path: str,
    mech_path: str,
    *,
    config: StoichIntervalTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    import cantera as ct

    cfg = config or StoichIntervalTrainingConfig()
    if cfg.model_variant not in {"stoich", "thermo-affinity", "substep", "substep-soft-thermo", "thermo-progress-substep"}:
        raise ValueError(f"Unsupported model_variant: {cfg.model_variant}")
    current, target, dt, _dt_bin, dt_bin_edges, species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
    log_dt = np.log(np.maximum(dt, 1e-300))[:, None]
    midpoint, midpoint_dt = load_interval_midpoint_arrays(source_path, dtype=np.float64)
    use_midpoint = cfg.midpoint_loss_weight > 0.0 and midpoint is not None and midpoint_dt is not None

    state_mean = current.mean(axis=0)
    state_std = current.std(axis=0)
    state_std = np.where(state_std > 0, state_std, 1.0)
    log_dt_mean = log_dt.mean(axis=0)
    log_dt_std = log_dt.std(axis=0)
    log_dt_std = np.where(log_dt_std > 0, log_dt_std, 1.0)

    current_norm = ((current - state_mean) / state_std).astype(np.float32)
    target_norm = ((target - state_mean) / state_std).astype(np.float32)
    log_dt_norm = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)
    if cfg.adaptive_substeps:
        substeps_np = np.ones_like(dt, dtype=np.int64)
        substeps_np = np.where(dt >= 1e-7, 2, substeps_np)
        substeps_np = np.where(dt >= 1e-5, 4, substeps_np)
        substeps_np = np.where(dt >= 1e-4, 8, substeps_np)
        log_dt_step_norm = _normalized_log_dt(dt / substeps_np, log_dt_mean, log_dt_std)
    else:
        substeps_np = np.full_like(dt, cfg.substeps, dtype=np.int64)
        log_dt_step_norm = _normalized_log_dt(dt / max(cfg.substeps, 1), log_dt_mean, log_dt_std)
    if use_midpoint:
        midpoint_norm = ((midpoint - state_mean) / state_std).astype(np.float32)
        midpoint_log_dt_norm = _normalized_log_dt(midpoint_dt, log_dt_mean, log_dt_std)
        if cfg.adaptive_substeps:
            midpoint_substeps_np = np.ones_like(midpoint_dt, dtype=np.int64)
            midpoint_substeps_np = np.where(midpoint_dt >= 1e-7, 2, midpoint_substeps_np)
            midpoint_substeps_np = np.where(midpoint_dt >= 1e-5, 4, midpoint_substeps_np)
            midpoint_substeps_np = np.where(midpoint_dt >= 1e-4, 8, midpoint_substeps_np)
            midpoint_log_dt_step_norm = _normalized_log_dt(midpoint_dt / midpoint_substeps_np, log_dt_mean, log_dt_std)
        else:
            midpoint_substeps_np = np.full_like(midpoint_dt, cfg.substeps, dtype=np.int64)
            midpoint_log_dt_step_norm = _normalized_log_dt(midpoint_dt / max(cfg.substeps, 1), log_dt_mean, log_dt_std)
    state_std_tensor = torch.tensor(state_std, dtype=torch.float64)
    thermo_feature_mean = None
    thermo_feature_std = None

    phase_name = _phase_from_attrs(attrs)
    gas = ct.Solution(mech_path, phase_name) if phase_name is not None else ct.Solution(mech_path)
    stoich_mass = stoichiometric_mass_fraction_matrix(gas)
    stoich = reaction_stoichiometry(gas)
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    tensor_items = [
        torch.tensor(current_norm, dtype=torch.float32),
        torch.tensor(target_norm, dtype=torch.float32),
        torch.tensor(current[:, 2:], dtype=torch.float64),
        torch.tensor(target[:, 2:], dtype=torch.float64),
        torch.tensor(log_dt_norm, dtype=torch.float32),
        torch.tensor(dt, dtype=torch.float64),
        torch.tensor(log_dt_step_norm, dtype=torch.float32),
        torch.tensor(substeps_np, dtype=torch.long),
    ]
    if cfg.model_variant in THERMO_FEATURE_VARIANTS:
        affinity_hat, log_reactant_activity, reversible = load_interval_thermo_arrays(source_path, dtype=np.float32)
        if affinity_hat.shape[1] != gas.n_reactions:
            raise ValueError("Thermo feature reaction count does not match mechanism")
        if not np.array_equal(reversible, stoich.reversible):
            raise ValueError("Thermo feature reaction reversibility does not match mechanism")
        if cfg.model_variant == "substep-soft-thermo":
            thermo_features_np = np.concatenate([affinity_hat, log_reactant_activity], axis=1).astype(np.float64)
            thermo_feature_mean = thermo_features_np.mean(axis=0)
            thermo_feature_std = thermo_features_np.std(axis=0)
            thermo_feature_std = np.where(thermo_feature_std > 0, thermo_feature_std, 1.0)
            thermo_features_np = (thermo_features_np - thermo_feature_mean) / thermo_feature_std
            if cfg.thermo_feature_clip > 0.0:
                thermo_features_np = np.clip(thermo_features_np, -cfg.thermo_feature_clip, cfg.thermo_feature_clip)
            affinity_hat = thermo_features_np[:, : gas.n_reactions].astype(np.float32)
            log_reactant_activity = thermo_features_np[:, gas.n_reactions :].astype(np.float32)
        tensor_items.extend(
            [
                torch.tensor(affinity_hat, dtype=torch.float32),
                torch.tensor(log_reactant_activity, dtype=torch.float32),
            ]
        )
    if use_midpoint:
        tensor_items.extend(
            [
                torch.tensor(midpoint_norm, dtype=torch.float32),
                torch.tensor(midpoint[:, 2:], dtype=torch.float64),
                torch.tensor(midpoint_log_dt_norm, dtype=torch.float32),
                torch.tensor(midpoint_log_dt_step_norm, dtype=torch.float32),
                torch.tensor(midpoint_substeps_np, dtype=torch.long),
            ]
        )
    tensors = torch.utils.data.TensorDataset(*tensor_items)
    loader = torch.utils.data.DataLoader(tensors, batch_size=cfg.batch_size, shuffle=True)
    if cfg.model_variant == "thermo-affinity":
        model = ThermoStoichiometricIntervalModel(
            input_dim=current.shape[-1],
            stoichiometric_mass_matrix=stoich_mass,
            reaction_reversible=stoich.reversible,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            extent_scale=cfg.thermo_extent_scale,
            force_scale=cfg.thermo_force_scale,
            availability_threshold=cfg.thermo_availability_threshold,
            availability_slope=cfg.thermo_availability_slope,
        ).to(torch_device)
    elif cfg.model_variant in {"substep", "substep-soft-thermo"}:
        model = LatentSubstepStoichiometricIntervalModel(
            input_dim=current.shape[-1],
            stoichiometric_mass_matrix=stoich_mass,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            substeps=cfg.substeps,
            flux_mode=cfg.flux_mode,
            transform_alpha=cfg.transform_alpha,
            transform_scale_by_alpha=cfg.transform_scale_by_alpha,
            thermo_input_dim=2 * gas.n_reactions if cfg.model_variant == "substep-soft-thermo" else 0,
            thermo_embedding_dim=cfg.thermo_embedding_dim,
            reaction_extent_raw_clip=cfg.reaction_extent_raw_clip,
        ).to(torch_device)
    elif cfg.model_variant == "thermo-progress-substep":
        if ThermoProgressSubstepStoichiometricIntervalModel is None:
            raise ImportError(
                "model_variant='thermo-progress-substep' requires "
                "ThermoProgressSubstepStoichiometricIntervalModel in dfode_kit.models.latent_baseline"
            )
        model = ThermoProgressSubstepStoichiometricIntervalModel(
            input_dim=current.shape[-1],
            stoichiometric_mass_matrix=stoich_mass,
            reaction_reversible=stoich.reversible,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            substeps=cfg.substeps,
            force_clip=cfg.thermo_progress_force_clip,
            mobility_scale=cfg.thermo_progress_mobility_scale,
            progress_mode=cfg.thermo_progress_mode,
            latent_update_mode=cfg.latent_update_mode,
            latent_step_scale=cfg.latent_step_scale,
            proximal_correction_steps=cfg.proximal_correction_steps,
            proximal_correction_lr=cfg.proximal_correction_lr,
            proximal_free_energy_weight=cfg.proximal_free_energy_weight,
            proximal_y_floor=cfg.proximal_y_floor,
            positivity_safety_factor=cfg.positivity_safety_factor,
        ).to(torch_device)
    else:
        model = StoichiometricIntervalModel(
            input_dim=current.shape[-1],
            stoichiometric_mass_matrix=stoich_mass,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            flux_mode=cfg.flux_mode,
            transform_alpha=cfg.transform_alpha,
            transform_scale_by_alpha=cfg.transform_scale_by_alpha,
        ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    perturb_generator = None
    if cfg.input_perturb_alpha > 0.0:
        perturb_generator = torch.Generator(device=torch_device)
        perturb_generator.manual_seed(cfg.input_perturb_seed)

    last_loss = float("nan")
    last_state_loss = float("nan")
    last_tp_loss = float("nan")
    last_y_loss = float("nan")
    last_delta_loss = float("nan")
    last_semigroup_loss = float("nan")
    last_midpoint_loss = float("nan")
    last_error_loss = float("nan")
    last_step_policy_loss = float("nan")

    state_mean_tensor = torch.tensor(state_mean, dtype=torch.float64, device=torch_device)
    state_std_tensor_device = torch.tensor(state_std, dtype=torch.float64, device=torch_device)

    train_start_time = time.time()
    for _epoch in range(cfg.epochs):
        epoch_start_time = time.time()
        for batch in loader:
            current_batch, target_batch, current_y_batch, target_y_batch, log_dt_batch, dt_batch, log_dt_step_batch, substeps_batch = batch[:8]
            current_batch = current_batch.to(torch_device)
            target_batch = target_batch.to(torch_device)
            current_y_batch = current_y_batch.to(torch_device)
            target_y_batch = target_y_batch.to(torch_device)
            log_dt_batch = log_dt_batch.to(torch_device)
            dt_batch = dt_batch.to(torch_device)
            log_dt_step_batch = log_dt_step_batch.to(torch_device)
            substeps_batch = substeps_batch.to(torch_device)
            affinity_batch = None
            log_activity_batch = None
            cursor = 8
            if cfg.model_variant in THERMO_FEATURE_VARIANTS:
                affinity_batch = batch[cursor].to(torch_device)
                log_activity_batch = batch[cursor + 1].to(torch_device)
                cursor += 2
            midpoint_batch = None
            midpoint_y_batch = None
            midpoint_log_dt_batch = None
            midpoint_log_dt_step_batch = None
            midpoint_substeps_batch = None
            if use_midpoint:
                midpoint_batch = batch[cursor].to(torch_device)
                midpoint_y_batch = batch[cursor + 1].to(torch_device)
                midpoint_log_dt_batch = batch[cursor + 2].to(torch_device)
                midpoint_log_dt_step_batch = batch[cursor + 3].to(torch_device)
                midpoint_substeps_batch = batch[cursor + 4].to(torch_device)
            if cfg.input_perturb_alpha > 0.0:
                current_physical_batch = torch.cat(
                    [
                        current_batch.to(torch.float64) * torch.tensor(state_std, dtype=torch.float64, device=torch_device)
                        + torch.tensor(state_mean, dtype=torch.float64, device=torch_device)
                    ],
                    dim=-1,
                )
                current_physical_batch = _perturb_current_state(
                    current_physical_batch,
                    state_std=state_std_tensor.to(torch_device),
                    alpha=cfg.input_perturb_alpha,
                    generator=perturb_generator,
                )
                current_batch = ((current_physical_batch - torch.tensor(state_mean, dtype=torch.float64, device=torch_device)) / torch.tensor(state_std, dtype=torch.float64, device=torch_device)).to(torch.float32)
                current_y_batch = current_physical_batch[..., 2:]
            output = _forward_interval_model(
                model,
                current_batch,
                log_dt_batch,
                current_y_batch,
                model_variant=cfg.model_variant,
                affinity_batch=affinity_batch,
                log_activity_batch=log_activity_batch,
                log_dt_step_batch=log_dt_step_batch,
                substeps_batch=substeps_batch if cfg.adaptive_substeps and cfg.model_variant in SUBSTEP_VARIANTS else None,
            )
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
            state_loss = tp_loss + y_loss + cfg.species_loss_weight * delta_loss
            semigroup_loss = torch.zeros((), dtype=torch.float64, device=torch_device)
            error_loss = torch.zeros((), dtype=torch.float64, device=torch_device)
            if cfg.semigroup_loss_weight > 0.0 and cfg.model_variant in SUBSTEP_VARIANTS:
                half_dt_batch = dt_batch / 2.0
                half_log_dt = torch.log(torch.clamp(half_dt_batch, min=1e-300))[:, None]
                half_log_dt_norm = ((half_log_dt - torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)) / torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)).to(torch.float32)
                if cfg.adaptive_substeps:
                    half_substeps_batch = _substeps_for_dt(half_dt_batch)
                    half_log_dt_step_norm = ((torch.log(torch.clamp(half_dt_batch / half_substeps_batch.to(torch.float64), min=1e-300))[:, None] - torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)) / torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)).to(torch.float32)
                else:
                    half_substeps_batch = torch.full_like(substeps_batch, cfg.substeps)
                    half_log_dt_step_norm = ((torch.log(torch.clamp(half_dt_batch / max(cfg.substeps, 1), min=1e-300))[:, None] - torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)) / torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)).to(torch.float32)
                first_half = _forward_interval_model(
                    model,
                    current_batch,
                    half_log_dt_norm,
                    current_y_batch,
                    model_variant=cfg.model_variant,
                    affinity_batch=affinity_batch,
                    log_activity_batch=log_activity_batch,
                    log_dt_step_batch=half_log_dt_step_norm,
                    substeps_batch=half_substeps_batch if cfg.adaptive_substeps else None,
                )
                second_input = _physical_to_normalized_state(
                    first_half["next_state"],
                    state_mean_tensor,
                    state_std_tensor_device,
                )
                second_half = _forward_interval_model(
                    model,
                    second_input,
                    half_log_dt_norm,
                    first_half["next_state"][..., 2:],
                    model_variant=cfg.model_variant,
                    affinity_batch=affinity_batch,
                    log_activity_batch=log_activity_batch,
                    log_dt_step_batch=half_log_dt_step_norm,
                    substeps_batch=half_substeps_batch if cfg.adaptive_substeps else None,
                )
                semigroup_loss = _regression_loss(
                    signed_power_transform(second_half["next_state"][..., 2:], alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha),
                    signed_power_transform(output["next_state"][..., 2:].detach(), alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha),
                    loss_kind=cfg.loss_kind,
                )
                state_loss = state_loss + cfg.semigroup_loss_weight * semigroup_loss
                if cfg.error_loss_weight > 0.0 and "error_estimate" in output:
                    error_target = torch.mean(
                        torch.abs(
                            signed_power_transform(second_half["next_state"][..., 2:].detach(), alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha)
                            - signed_power_transform(output["next_state"][..., 2:].detach(), alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha)
                        ),
                        dim=-1,
                        keepdim=True,
                    )
                    error_loss = _regression_loss(output["error_estimate"], error_target, loss_kind="mae")
                    state_loss = state_loss + cfg.error_loss_weight * error_loss
            midpoint_loss = torch.zeros((), dtype=torch.float64, device=torch_device)
            if use_midpoint and cfg.model_variant in SUBSTEP_VARIANTS:
                mid_output = _forward_interval_model(
                    model,
                    current_batch,
                    midpoint_log_dt_batch,
                    current_y_batch,
                    model_variant=cfg.model_variant,
                    affinity_batch=affinity_batch,
                    log_activity_batch=log_activity_batch,
                    log_dt_step_batch=midpoint_log_dt_step_batch,
                    substeps_batch=midpoint_substeps_batch if cfg.adaptive_substeps else None,
                )
                midpoint_loss = _regression_loss(
                    signed_power_transform(mid_output["next_state"][..., 2:], alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha),
                    signed_power_transform(midpoint_y_batch, alpha=cfg.transform_alpha, scale_by_alpha=cfg.transform_scale_by_alpha),
                    loss_kind=cfg.loss_kind,
                )
                state_loss = state_loss + cfg.midpoint_loss_weight * midpoint_loss
            step_policy_loss = torch.zeros((), dtype=torch.float64, device=torch_device)
            if cfg.step_policy_loss_weight > 0.0 and cfg.model_variant in SUBSTEP_VARIANTS and "step_policy_logits" in output:
                target_substeps = _substeps_for_dt(dt_batch)
                target_labels = _substep_labels(target_substeps)
                step_policy_loss = torch.nn.functional.cross_entropy(output["step_policy_logits"], target_labels).to(torch.float64)
                state_loss = state_loss + cfg.step_policy_loss_weight * step_policy_loss

            if not torch.isfinite(state_loss):
                if cfg.skip_nonfinite_batches:
                    continue
                raise FloatingPointError("Non-finite training loss")
            optimizer.zero_grad()
            state_loss.backward()
            if cfg.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()

            last_loss = float(state_loss.detach().cpu())
            last_state_loss = float(state_loss.detach().cpu())
            last_tp_loss = float(tp_loss.detach().cpu())
            last_y_loss = float(y_loss.detach().cpu())
            last_delta_loss = float(delta_loss.detach().cpu())
            last_semigroup_loss = float(semigroup_loss.detach().cpu())
            last_midpoint_loss = float(midpoint_loss.detach().cpu())
            last_error_loss = float(error_loss.detach().cpu())
            last_step_policy_loss = float(step_policy_loss.detach().cpu())
        epoch_number = _epoch + 1
        if cfg.log_every_epochs > 0 and (epoch_number == 1 or epoch_number % cfg.log_every_epochs == 0 or epoch_number == cfg.epochs):
            elapsed = time.time() - train_start_time
            epoch_seconds = time.time() - epoch_start_time
            eta = elapsed / epoch_number * (cfg.epochs - epoch_number) if epoch_number > 0 else float("nan")
            print(
                "epoch "
                f"{epoch_number}/{cfg.epochs} "
                f"loss={last_loss:.6e} "
                f"tp={last_tp_loss:.6e} "
                f"y={last_y_loss:.6e} "
                f"delta={last_delta_loss:.6e} "
                f"semigroup={last_semigroup_loss:.6e} "
                f"midpoint={last_midpoint_loss:.6e} "
                f"error={last_error_loss:.6e} "
                f"step_policy={last_step_policy_loss:.6e} "
                f"epoch_seconds={epoch_seconds:.2f} "
                f"elapsed_minutes={elapsed / 60.0:.2f} "
                f"eta_minutes={eta / 60.0:.2f}",
                flush=True,
            )

    if cfg.model_variant == "thermo-affinity":
        model_type = "thermo_stoichiometric_interval"
    elif cfg.model_variant in {"substep", "substep-soft-thermo"}:
        model_type = "latent_substep_stoichiometric_interval"
    elif cfg.model_variant == "thermo-progress-substep":
        model_type = "thermo_progress_substep_stoichiometric_interval"
    else:
        model_type = "stoichiometric_interval"
    torch.save(
        {
            "model_type": model_type,
            "net": model.state_dict(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "log_dt_mean": log_dt_mean.tolist(),
            "log_dt_std": log_dt_std.tolist(),
            "species_names": species_names,
            "mechanism": mech_path,
            "phase_name": phase_name,
            "stoichiometric_mass_matrix": stoich_mass.tolist(),
            "reaction_reversible": stoich.reversible.astype(int).tolist(),
            "thermo_feature_mean": None if thermo_feature_mean is None else thermo_feature_mean.tolist(),
            "thermo_feature_std": None if thermo_feature_std is None else thermo_feature_std.tolist(),
            "dt_bin_edges": dt_bin_edges.tolist(),
            "training_config": {
                "model_variant": cfg.model_variant,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "transform_alpha": cfg.transform_alpha,
                "species_loss_weight": cfg.species_loss_weight,
                "loss_kind": cfg.loss_kind,
                "transform_scale_by_alpha": cfg.transform_scale_by_alpha,
                "flux_mode": cfg.flux_mode,
                "input_perturb_alpha": cfg.input_perturb_alpha,
                "input_perturb_seed": cfg.input_perturb_seed,
                "thermo_extent_scale": cfg.thermo_extent_scale,
                "thermo_force_scale": cfg.thermo_force_scale,
                "thermo_availability_threshold": cfg.thermo_availability_threshold,
                "thermo_availability_slope": cfg.thermo_availability_slope,
                "substeps": cfg.substeps,
                "adaptive_substeps": cfg.adaptive_substeps,
                "semigroup_loss_weight": cfg.semigroup_loss_weight,
                "midpoint_loss_weight": cfg.midpoint_loss_weight,
                "thermo_embedding_dim": cfg.thermo_embedding_dim,
                "log_every_epochs": cfg.log_every_epochs,
                "gradient_clip_norm": cfg.gradient_clip_norm,
                "reaction_extent_raw_clip": cfg.reaction_extent_raw_clip,
                "thermo_feature_clip": cfg.thermo_feature_clip,
                "skip_nonfinite_batches": cfg.skip_nonfinite_batches,
                "error_loss_weight": cfg.error_loss_weight,
                "step_policy_loss_weight": cfg.step_policy_loss_weight,
                "learned_adaptive_substeps": cfg.learned_adaptive_substeps,
                "thermo_progress_force_clip": cfg.thermo_progress_force_clip,
                "thermo_progress_mobility_scale": cfg.thermo_progress_mobility_scale,
                "thermo_progress_mode": cfg.thermo_progress_mode,
                "latent_update_mode": cfg.latent_update_mode,
                "latent_step_scale": cfg.latent_step_scale,
                "proximal_correction_steps": cfg.proximal_correction_steps,
                "proximal_correction_lr": cfg.proximal_correction_lr,
                "proximal_free_energy_weight": cfg.proximal_free_energy_weight,
                "proximal_y_floor": cfg.proximal_y_floor,
                "positivity_safety_factor": cfg.positivity_safety_factor,
            },
            "final_metrics": {
                "loss": last_loss,
                "state_loss": last_state_loss,
                "tp_loss": last_tp_loss,
                "y_loss": last_y_loss,
                "delta_loss": last_delta_loss,
                "semigroup_loss": last_semigroup_loss,
                "midpoint_loss": last_midpoint_loss,
                "error_loss": last_error_loss,
                "step_policy_loss": last_step_policy_loss,
            },
        },
        output_path,
    )

    return {
        "loss": last_loss,
        "state_loss": last_state_loss,
        "tp_loss": last_tp_loss,
        "y_loss": last_y_loss,
        "delta_loss": last_delta_loss,
        "semigroup_loss": last_semigroup_loss,
        "midpoint_loss": last_midpoint_loss,
        "error_loss": last_error_loss,
        "step_policy_loss": last_step_policy_loss,
    }
