#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from dfode_kit.evaluation.positive_interval import _load_positive_model
from dfode_kit.evaluation.stoich_interval import _load_model
from dfode_kit.models.positive_interval import (
    DELTA_H_TOTAL_MODE,
    DELTA_TEMPERATURE_MODE,
    TOTAL_ENTHALPY_TARGET_IDENTITY,
    TOTAL_ENTHALPY_TARGET_SIGNED_POWER,
    TOTAL_ENTHALPY_RESIDUAL_NONE,
    TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER,
    THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH,
    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
    THERMOCHEMICAL_HEAD_INPUT_LEGACY,
    THERMOCHEMICAL_HEAD_INPUT_STATE_TIME,
    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
    THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS,
    THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER,
    NeuralPatankarIntervalModel,
    inverse_signed_power,
)
from dfode_kit.physics.atom_conservation import reaction_stoichiometry


PREDICTED_DELTA_T_CLOSURE = "predicted-delta-temperature"
FLUENT_CONSTANT_ENTHALPY_CLOSURE = "fluent-constant-mixture-enthalpy"
FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE = (
    "predicted-fluent-total-enthalpy-increment"
)
THERMOCHEMICAL_CLOSURES = (
    PREDICTED_DELTA_T_CLOSURE,
    FLUENT_CONSTANT_ENTHALPY_CLOSURE,
    FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FluentReactionChannelWrapper(torch.nn.Module):
    """Map physical CFD inputs to transformed reaction channels."""

    def __init__(
        self,
        model: torch.nn.Module,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        log_dt_mean: float,
        log_dt_std: float,
    ):
        super().__init__()
        self.encoder = model.encoder
        self.flux_head = model.flux_head
        self.register_buffer(
            "state_mean",
            torch.as_tensor(state_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "state_std",
            torch.as_tensor(state_std, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_mean",
            torch.tensor(log_dt_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_std",
            torch.tensor(log_dt_std, dtype=torch.float32),
        )

    def forward(self, physical_input: torch.Tensor) -> torch.Tensor:
        state = physical_input[:, :-1]
        dt = torch.clamp(physical_input[:, -1:], min=1.0e-30)
        state_normalized = (state - self.state_mean) / self.state_std
        log_dt_normalized = (
            torch.log(dt) - self.log_dt_mean
        ) / self.log_dt_std
        latent = self.encoder(
            torch.cat([state_normalized, log_dt_normalized], dim=-1)
        )
        return self.flux_head(latent)


class FluentNeuralPatankarWrapper(torch.nn.Module):
    """Map physical CFD inputs directly to positive conservative delta_Y."""

    def __init__(
        self,
        model: torch.nn.Module,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        log_dt_mean: float,
        log_dt_std: float,
        include_temperature: bool = True,
        include_total_enthalpy_increment: bool = False,
        availability_mode: str = "hard",
        availability_p_norm: float = 32.0,
        availability_floor: float = 1e-30,
    ):
        super().__init__()
        if availability_mode not in {"hard", "smooth-safe"}:
            raise ValueError(
                "availability_mode must be 'hard' or 'smooth-safe'"
            )
        if availability_p_norm < 1.0:
            raise ValueError("availability_p_norm must be at least one")
        if include_temperature and include_total_enthalpy_increment:
            raise ValueError(
                "temperature and total-enthalpy outputs are mutually exclusive"
            )
        self.include_temperature = include_temperature
        self.include_total_enthalpy_increment = include_total_enthalpy_increment
        self.thermochemical_head_input_mode = str(
            getattr(
                model,
                "thermochemical_head_input_mode",
                THERMOCHEMICAL_HEAD_INPUT_LEGACY,
            )
        )
        self.thermochemical_aux_temperature = bool(
            getattr(model, "thermochemical_aux_temperature", False)
        )
        self.total_enthalpy_head_hidden_dim = int(
            getattr(model, "total_enthalpy_head_hidden_dim", 0)
        )
        self.total_enthalpy_head_depth = int(
            getattr(model, "total_enthalpy_head_depth", 0)
        )
        self.total_enthalpy_target_transform = str(
            getattr(
                model,
                "total_enthalpy_target_transform",
                TOTAL_ENTHALPY_TARGET_IDENTITY,
            )
        )
        self.total_enthalpy_transform_alpha = float(
            getattr(model, "total_enthalpy_transform_alpha", 0.1)
        )
        self.total_enthalpy_residual_mode = str(
            getattr(
                model,
                "total_enthalpy_residual_mode",
                TOTAL_ENTHALPY_RESIDUAL_NONE,
            )
        )
        self.total_enthalpy_residual_alpha = float(
            getattr(model, "total_enthalpy_residual_alpha", 0.1)
        )
        self.thermochemical_delta_y_feature_mode = str(
            getattr(
                model,
                "thermochemical_delta_y_feature_mode",
                THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
            )
        )
        self.thermochemical_process_feature_mode = str(
            getattr(
                model,
                "thermochemical_process_feature_mode",
                THERMOCHEMICAL_PROCESS_FEATURE_NONE,
            )
        )
        self.register_buffer(
            "thermochemical_delta_y_scales",
            getattr(
                model,
                "thermochemical_delta_y_scales",
                torch.ones(
                    model.process_stoich.shape[0], dtype=torch.float64
                ),
            ).detach().clone().to(torch.float64),
        )
        self.thermochemical_state_high_low = bool(
            getattr(model, "thermochemical_state_high_low", False)
        )
        self.thermochemical_state_low_scale = float(
            getattr(model, "thermochemical_state_low_scale", float(2**24))
        )
        self.availability_mode = availability_mode
        self.availability_p_norm = float(availability_p_norm)
        self.availability_floor = float(availability_floor)
        self.encoder = model.encoder
        self.demand_head = model.demand_head
        self.register_buffer(
            "process_gate",
            model.process_gate.detach().clone(),
        )
        if include_total_enthalpy_increment:
            self.thermochemical_head = model.total_enthalpy_delta_head
            if (
                self.total_enthalpy_residual_mode
                == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
            ):
                self.total_enthalpy_residual_head = (
                    model.total_enthalpy_residual_head
                )
            self.thermochemical_scale = float(model.total_enthalpy_delta_scale)
            if self.thermochemical_aux_temperature:
                self.thermochemical_aux_temperature_head = (
                    model.thermochemical_aux_temperature_head
                )
                self.auxiliary_temperature_scale = float(
                    model.temperature_delta_scale
                )
        else:
            self.thermochemical_head = model.temperature_delta_head
            self.thermochemical_scale = float(model.temperature_delta_scale)
        self.extent_scale = float(model.extent_scale)
        self.register_buffer("consumption", model.consumption.detach().clone())
        self.register_buffer(
            "process_stoich",
            model.process_stoich.detach().clone(),
        )
        self.register_buffer(
            "process_turnover_stoich",
            (
                2.0 * model.consumption.detach().clone()
                + model.process_stoich.detach().clone()
            ),
        )
        process_consumption = model.consumption.detach().clone().T
        reactant_counts = (process_consumption > 0.0).sum(dim=1)
        max_reactants = int(reactant_counts.max().item())
        reactant_indices = torch.zeros(
            (process_consumption.shape[0], max_reactants),
            dtype=torch.int64,
        )
        reactant_mask = torch.zeros_like(reactant_indices, dtype=torch.bool)
        for process_index in range(process_consumption.shape[0]):
            indices = torch.nonzero(
                process_consumption[process_index] > 0.0,
                as_tuple=False,
            ).flatten()
            reactant_indices[process_index, : indices.numel()] = indices
            reactant_mask[process_index, : indices.numel()] = True
        self.register_buffer("reactant_indices", reactant_indices)
        self.register_buffer("reactant_mask", reactant_mask)
        self.register_buffer(
            "state_mean",
            torch.as_tensor(state_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "state_std",
            torch.as_tensor(state_std, dtype=torch.float32),
        )
        self.register_buffer(
            "state_mean_float64",
            torch.as_tensor(state_mean, dtype=torch.float64),
        )
        self.register_buffer(
            "state_std_float64",
            torch.as_tensor(state_std, dtype=torch.float64),
        )
        self.register_buffer(
            "log_dt_mean",
            torch.tensor(log_dt_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_std",
            torch.tensor(log_dt_std, dtype=torch.float32),
        )

    def _normalize_state(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.thermochemical_state_high_low:
            normalized_float64 = (
                state.to(torch.float64) - self.state_mean_float64
            ) / self.state_std_float64
            normalized_high = normalized_float64.to(torch.float32)
            normalized_low = (
                (
                    normalized_float64
                    - normalized_high.to(torch.float64)
                )
                * self.thermochemical_state_low_scale
            ).to(torch.float32)
            return normalized_high, normalized_low
        normalized_high = (
            (state.to(torch.float64) - self.state_mean_float64)
            / self.state_std_float64
        ).to(torch.float32)
        return normalized_high, torch.empty(
            (state.shape[0], 0),
            dtype=torch.float32,
            device=state.device,
        )

    def forward(
        self,
        physical_input: torch.Tensor,
        current_species: torch.Tensor,
    ) -> torch.Tensor:
        state = physical_input[:, :-1]
        dt = torch.clamp(physical_input[:, -1:], min=1.0e-30)
        state_normalized, thermochemical_state_low = self._normalize_state(
            state
        )
        log_dt_normalized = (
            torch.log(dt) - self.log_dt_mean
        ) / self.log_dt_std
        hidden = self.encoder(
            torch.cat([state_normalized, log_dt_normalized], dim=-1)
        )
        demand = torch.nn.functional.softplus(
            self.demand_head(hidden).to(torch.float64)
        ) * self.extent_scale * self.process_gate
        current_species = current_species.to(torch.float64)
        requested = demand @ self.consumption.T
        availability = torch.clamp(
            current_species / requested.clamp_min(self.availability_floor),
            min=0.0,
            max=1.0,
        )
        reactant_availability = availability[:, self.reactant_indices]
        candidates = torch.where(
            self.reactant_mask[None],
            reactant_availability,
            reactant_availability * 0.0 + 1.0,
        )
        hard_availability = candidates.amin(dim=-1)
        if self.availability_mode == "smooth-safe":
            inverse_log = -torch.log(
                reactant_availability.clamp_min(
                    torch.finfo(torch.float64).tiny
                )
            )
            terms = self.availability_p_norm * inverse_log
            terms = torch.where(
                self.reactant_mask[None],
                terms,
                torch.full_like(terms, -torch.inf),
            )
            has_reactant = self.reactant_mask.any(dim=-1)[None]
            log_upper_inverse = torch.logsumexp(terms, dim=-1) / self.availability_p_norm
            smooth_availability = torch.where(
                has_reactant,
                torch.exp(-log_upper_inverse),
                torch.ones_like(log_upper_inverse),
            )
            smooth_availability = torch.where(
                has_reactant,
                smooth_availability
                * (1.0 - 8.0 * torch.finfo(torch.float64).eps),
                torch.ones_like(smooth_availability),
            )
            process_availability = torch.where(
                smooth_availability > hard_availability,
                hard_availability,
                smooth_availability,
            ).clamp(0.0, 1.0)
        else:
            process_availability = hard_availability
        process_extent = (
            demand * process_availability * (1.0 - 1.0e-12)
        )
        delta_y = process_extent @ self.process_stoich.T
        thermochemical_process_features = None
        if (
            self.include_total_enthalpy_increment
            and self.thermochemical_process_feature_mode
            == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
        ):
            species_gross_turnover = (
                process_extent @ self.process_turnover_stoich.T
            )
            thermochemical_process_features = torch.pow(
                species_gross_turnover.clamp_min(0.0), 0.1
            ).to(hidden.dtype)
        elif (
            self.include_total_enthalpy_increment
            and self.thermochemical_process_feature_mode
            == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
        ):
            transformed_process_extent = torch.pow(
                process_extent.clamp_min(1.0e-30), 0.1
            )
            thermochemical_process_features = torch.where(
                process_extent == 0.0,
                torch.zeros_like(transformed_process_extent),
                transformed_process_extent,
            ).to(hidden.dtype)
        if not (
            self.include_temperature or self.include_total_enthalpy_increment
        ):
            return delta_y
        if (
            self.thermochemical_delta_y_feature_mode
            == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
        ):
            temperature_species_features = torch.asinh(
                delta_y / self.thermochemical_delta_y_scales
            ).to(hidden.dtype)
        else:
            temperature_species_features = (
                torch.sign(delta_y)
                * torch.pow(torch.abs(delta_y).clamp_min(1.0e-30), 0.1)
            ).to(hidden.dtype)
        thermochemical_feature_blocks = [hidden]
        if (
            self.thermochemical_head_input_mode
            == THERMOCHEMICAL_HEAD_INPUT_STATE_TIME
        ):
            thermochemical_feature_blocks.extend(
                [state_normalized, log_dt_normalized]
            )
        thermochemical_feature_blocks.append(temperature_species_features)
        thermochemical_features = torch.cat(
            thermochemical_feature_blocks, dim=1
        )
        if self.thermochemical_aux_temperature:
            auxiliary_features = thermochemical_features
            if self.thermochemical_state_high_low:
                auxiliary_features = torch.cat(
                    [auxiliary_features, thermochemical_state_low], dim=1
                )
            auxiliary_delta_temperature_scaled = (
                self.thermochemical_aux_temperature_head(
                    auxiliary_features
                )
            )
            thermochemical_features = torch.cat(
                [thermochemical_features, auxiliary_delta_temperature_scaled],
                dim=1,
            )
        if self.thermochemical_state_high_low:
            thermochemical_features = torch.cat(
                [thermochemical_features, thermochemical_state_low], dim=1
            )
        if thermochemical_process_features is not None:
            thermochemical_features = torch.cat(
                [thermochemical_features, thermochemical_process_features],
                dim=1,
            )
        thermochemical_head_space = self.thermochemical_head(
            thermochemical_features
        ).to(torch.float64)
        if (
            self.include_total_enthalpy_increment
            and self.total_enthalpy_residual_mode
            == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
        ):
            residual_head_space = self.total_enthalpy_residual_head(
                thermochemical_features
            ).to(torch.float64)
            thermochemical_scaled = (
                thermochemical_head_space
                + inverse_signed_power(
                    residual_head_space,
                    self.total_enthalpy_residual_alpha,
                )
            )
        elif (
            self.include_total_enthalpy_increment
            and self.total_enthalpy_target_transform
            == TOTAL_ENTHALPY_TARGET_SIGNED_POWER
        ):
            thermochemical_scaled = inverse_signed_power(
                thermochemical_head_space,
                self.total_enthalpy_transform_alpha,
            )
        else:
            thermochemical_scaled = thermochemical_head_space
        thermochemical_increment = (
            thermochemical_scaled * self.thermochemical_scale
        )
        return torch.cat([thermochemical_increment, delta_y], dim=1)

    def dense_reference(
        self,
        physical_input: torch.Tensor,
        current_species: torch.Tensor,
    ) -> torch.Tensor:
        state = physical_input[:, :-1]
        dt = torch.clamp(physical_input[:, -1:], min=1.0e-30)
        state_normalized, _thermochemical_state_low = self._normalize_state(
            state
        )
        log_dt_normalized = (
            torch.log(dt) - self.log_dt_mean
        ) / self.log_dt_std
        hidden = self.encoder(
            torch.cat([state_normalized, log_dt_normalized], dim=-1)
        )
        demand = torch.nn.functional.softplus(
            self.demand_head(hidden).to(torch.float64)
        ) * self.extent_scale * self.process_gate
        current_species = current_species.to(torch.float64)
        requested = demand @ self.consumption.T
        availability = torch.clamp(
            current_species / requested.clamp_min(self.availability_floor),
            min=0.0,
            max=1.0,
        )
        reactant_mask = self.consumption.T[None] > 0.0
        candidates = torch.where(
            reactant_mask,
            availability[:, None, :],
            torch.ones(
                (),
                dtype=torch.float64,
                device=availability.device,
            ),
        )
        hard_availability = candidates.amin(dim=-1)
        if self.availability_mode == "smooth-safe":
            inverse_log = -torch.log(
                availability[:, None, :].clamp_min(
                    torch.finfo(torch.float64).tiny
                )
            )
            terms = self.availability_p_norm * inverse_log
            terms = torch.where(
                reactant_mask,
                terms,
                torch.full_like(terms, -torch.inf),
            )
            has_reactant = reactant_mask.any(dim=-1)
            log_upper_inverse = torch.logsumexp(terms, dim=-1) / self.availability_p_norm
            smooth_availability = torch.where(
                has_reactant,
                torch.exp(-log_upper_inverse),
                torch.ones_like(log_upper_inverse),
            )
            smooth_availability = torch.where(
                has_reactant,
                smooth_availability
                * (1.0 - 8.0 * torch.finfo(torch.float64).eps),
                torch.ones_like(smooth_availability),
            )
            process_availability = torch.where(
                smooth_availability > hard_availability,
                hard_availability,
                smooth_availability,
            ).clamp(0.0, 1.0)
        else:
            process_availability = hard_availability
        process_extent = (
            demand * process_availability * (1.0 - 1.0e-12)
        )
        return process_extent @ self.process_stoich.T


def _reference_inputs(
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: float,
    log_dt_std: float,
    *,
    use_float64: bool = False,
) -> np.ndarray:
    offsets = np.asarray([-0.15, -0.05, 0.05, 0.15], dtype=np.float64)
    states = state_mean[None, :] + offsets[:, None] * state_std[None, :]
    dt = np.exp(log_dt_mean + offsets * log_dt_std)[:, None]
    dtype = np.float64 if use_float64 else np.float32
    return np.concatenate([states, dt], axis=1).astype(dtype)


def _availability_settings(checkpoint: dict) -> tuple[str, float, float]:
    training_config = checkpoint.get("training_config", {})
    mode = str(
        checkpoint.get(
            "availability_mode",
            training_config.get("availability_mode", "hard"),
        )
    )
    p_norm = float(
        checkpoint.get(
            "availability_p_norm",
            training_config.get("availability_p_norm", 32.0),
        )
    )
    floor = float(training_config.get("positivity_floor", 1e-30))
    if mode not in {"hard", "smooth-safe"}:
        raise ValueError(f"Unsupported availability mode {mode!r}")
    if p_norm < 1.0:
        raise ValueError("availability_p_norm must be at least one")
    return mode, p_norm, floor


def _checkpoint_thermochemical_output_mode(checkpoint: dict) -> str:
    training_config = checkpoint.get("training_config", {})
    mode = str(
        checkpoint.get(
            "thermochemical_output_mode",
            training_config.get(
                "thermochemical_output_mode",
                DELTA_TEMPERATURE_MODE,
            ),
        )
    )
    if mode not in {DELTA_TEMPERATURE_MODE, DELTA_H_TOTAL_MODE}:
        raise ValueError(
            f"Unsupported checkpoint thermochemical output mode {mode!r}"
        )
    return mode


def _load_export_positive_model(checkpoint, gas, input_dim, device):
    mode = _checkpoint_thermochemical_output_mode(checkpoint)
    if mode == DELTA_TEMPERATURE_MODE:
        return _load_positive_model(checkpoint, gas, input_dim, device)
    cfg = checkpoint["training_config"]
    stoich = reaction_stoichiometry(gas)
    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    availability_mode, availability_p_norm, availability_floor = (
        _availability_settings(checkpoint)
    )
    model = NeuralPatankarIntervalModel(
        input_dim=input_dim,
        reactant_mass_matrix=molecular_weights[:, None] * stoich.reactants,
        product_mass_matrix=molecular_weights[:, None] * stoich.products,
        reaction_reversible=stoich.reversible,
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        extent_scale=float(cfg["extent_scale"]),
        availability_floor=availability_floor,
        thermochemical_output_mode=DELTA_H_TOTAL_MODE,
        thermochemical_head_input_mode=str(
            cfg.get(
                "thermochemical_head_input_mode",
                THERMOCHEMICAL_HEAD_INPUT_LEGACY,
            )
        ),
        thermochemical_head_hidden_dim=(
            int(cfg.get("thermochemical_head_hidden_dim", 0)) or None
        ),
        thermochemical_head_depth=int(
            cfg.get("thermochemical_head_depth", 1)
        ),
        total_enthalpy_head_hidden_dim=int(
            cfg.get("total_enthalpy_head_hidden_dim", 0)
        ),
        total_enthalpy_head_depth=int(
            cfg.get("total_enthalpy_head_depth", 0)
        ),
        thermochemical_aux_temperature=bool(
            cfg.get("thermochemical_aux_temperature", False)
        ),
        thermochemical_process_feature_mode=str(
            cfg.get(
                "thermochemical_process_feature_mode",
                THERMOCHEMICAL_PROCESS_FEATURE_NONE,
            )
        ),
        thermochemical_state_high_low=bool(
            cfg.get("thermochemical_state_high_low", False)
        ),
        thermochemical_state_low_scale=float(
            cfg.get("thermochemical_state_low_scale", float(2**24))
        ),
          total_enthalpy_delta_scale=float(
            checkpoint.get(
                "total_enthalpy_delta_scale",
                cfg.get("total_enthalpy_delta_scale", 1.0e3),
            )
          ),
          total_enthalpy_target_transform=str(
              cfg.get(
                  "total_enthalpy_target_transform",
                  TOTAL_ENTHALPY_TARGET_IDENTITY,
              )
          ),
          total_enthalpy_transform_alpha=float(
              cfg.get("total_enthalpy_transform_alpha", 0.1)
          ),
          total_enthalpy_residual_mode=str(
              cfg.get(
                  "total_enthalpy_residual_mode",
                  TOTAL_ENTHALPY_RESIDUAL_NONE,
              )
          ),
          total_enthalpy_residual_alpha=float(
              cfg.get("total_enthalpy_residual_alpha", 0.1)
          ),
        availability_mode=availability_mode,
        availability_p_norm=availability_p_norm,
    )
    model.load_state_dict(checkpoint["net"], strict=True)
    return model.to(device).eval()


def export_artifact(
    checkpoint_path: Path,
    mechanism_path: Path,
    output_dir: Path,
    thermochemical_closure: str | None = None,
) -> dict:
    if (
        thermochemical_closure is not None
        and thermochemical_closure not in THERMOCHEMICAL_CLOSURES
    ):
        raise ValueError(
            "Unsupported thermochemical closure "
            f"{thermochemical_closure!r}; expected one of "
            f"{THERMOCHEMICAL_CLOSURES}"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_type = checkpoint.get("model_type", "stoichiometric_interval")
    positive_model_type = checkpoint.get("positive_model_type")
    is_neural_patankar = positive_model_type == "neural-patankar"
    checkpoint_thermochemical_mode = _checkpoint_thermochemical_output_mode(
        checkpoint
    )
    if thermochemical_closure is None:
        thermochemical_closure = (
            FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE
            if checkpoint_thermochemical_mode == DELTA_H_TOTAL_MODE
            else PREDICTED_DELTA_T_CLOSURE
        )
    if (
        thermochemical_closure == PREDICTED_DELTA_T_CLOSURE
        and checkpoint_thermochemical_mode != DELTA_TEMPERATURE_MODE
    ):
        raise ValueError(
            "predicted-delta-temperature export requires a legacy "
            "delta-temperature checkpoint"
        )
    if (
        thermochemical_closure == FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE
        and checkpoint_thermochemical_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "total-enthalpy-increment export requires a delta-h-total checkpoint"
        )
    predicts_temperature = (
        is_neural_patankar
        and thermochemical_closure == PREDICTED_DELTA_T_CLOSURE
    )
    predicts_total_enthalpy_increment = (
        is_neural_patankar
        and thermochemical_closure
        == FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE
    )
    emits_thermochemical_scalar = (
        predicts_temperature or predicts_total_enthalpy_increment
    )
    effective_thermochemical_closure = thermochemical_closure
    if model_type != "stoichiometric_interval" and not is_neural_patankar:
        raise ValueError(
            "The Fluent exporter supports stoichiometric_interval and "
            f"neural-patankar checkpoints, got {model_type!r}/"
            f"{positive_model_type!r}"
        )

    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    log_dt_mean = float(np.asarray(checkpoint["log_dt_mean"]).reshape(-1)[0])
    log_dt_std = float(np.asarray(checkpoint["log_dt_std"]).reshape(-1)[0])
    if log_dt_std <= 1.0e-8:
        log_dt_std = 1.0

    species_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in checkpoint["species_names"]
    ]
    if is_neural_patankar:
        import cantera as ct

        gas = ct.Solution(str(mechanism_path))
        stoichiometry = np.asarray(
            checkpoint["molecular_weights"],
            dtype=np.float64,
        )[:, None] * np.asarray(
            checkpoint["stoichiometric_matrix"],
            dtype=np.float64,
        )
    else:
        stoichiometry = np.asarray(
            checkpoint["stoichiometric_mass_matrix"],
            dtype=np.float64,
            order="C",
        )
    n_species, n_reactions = stoichiometry.shape
    if state_mean.size != n_species + 2:
        raise ValueError("Checkpoint state and stoichiometry dimensions disagree")
    if len(species_names) != n_species:
        raise ValueError("Checkpoint species names and stoichiometry disagree")

    if is_neural_patankar:
        model = _load_export_positive_model(
            checkpoint,
            gas,
            state_mean.size,
            torch.device("cpu"),
        )
        availability_mode, availability_p_norm, availability_floor = (
            _availability_settings(checkpoint)
        )
        wrapper = FluentNeuralPatankarWrapper(
            model,
            state_mean,
            state_std,
            log_dt_mean,
            log_dt_std,
            include_temperature=predicts_temperature,
            include_total_enthalpy_increment=(
                predicts_total_enthalpy_increment
            ),
            availability_mode=availability_mode,
            availability_p_norm=availability_p_norm,
            availability_floor=availability_floor,
        ).eval()
        if predicts_temperature:
            output_mode = "direct_delta_t_delta_y"
        elif predicts_total_enthalpy_increment:
            output_mode = "direct_delta_h_total_delta_y"
        else:
            output_mode = "direct_delta_y"
        runtime_reaction_width = n_species + int(emits_thermochemical_scalar)
    else:
        model = _load_model(
            checkpoint,
            state_mean.size,
            torch.device("cpu"),
        )
        wrapper = FluentReactionChannelWrapper(
            model,
            state_mean,
            state_std,
            log_dt_mean,
            log_dt_std,
        ).eval()
        output_mode = "reaction_channel"
        runtime_reaction_width = n_reactions

    thermochemical_state_high_low = bool(
        is_neural_patankar
        and getattr(wrapper, "thermochemical_state_high_low", False)
    )
    reference_input = _reference_inputs(
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
        use_float64=thermochemical_state_high_low,
    )
    reference_tensor = torch.from_numpy(reference_input)
    reference_species = torch.from_numpy(
        np.asarray(reference_input[:, 2:-1], dtype=np.float64)
    )
    with torch.inference_mode():
        reference_output = (
            wrapper(reference_tensor, reference_species).cpu().numpy()
            if is_neural_patankar
            else wrapper(reference_tensor).cpu().numpy()
        )
        sparse_equivalence_max_abs_error = (
            float(
                np.max(
                    np.abs(
                        (
                            reference_output[:, 1:]
                            if emits_thermochemical_scalar
                            else reference_output
                        )
                        - wrapper.dense_reference(
                            reference_tensor,
                            reference_species,
                        ).cpu().numpy()
                    ),
                    initial=0.0,
                )
            )
            if is_neural_patankar
            else 0.0
        )
        trace_input = (
            (reference_tensor, reference_species)
            if is_neural_patankar
            else reference_tensor
        )
        traced = torch.jit.trace(wrapper, trace_input, strict=True)
        traced_output = (
            traced(reference_tensor, reference_species).cpu().numpy()
            if is_neural_patankar
            else traced(reference_tensor).cpu().numpy()
        )
    if sparse_equivalence_max_abs_error > 1.0e-14:
        raise RuntimeError(
            "Sparse Patankar limiter differs from dense reference: "
            f"max_abs_error={sparse_equivalence_max_abs_error}"
        )
    max_abs_error = float(
        np.max(np.abs(reference_output - traced_output), initial=0.0)
    )
    if max_abs_error > 1.0e-6:
        raise RuntimeError(
            f"TorchScript export mismatch: max_abs_error={max_abs_error}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    traced.save(str(model_path))
    mechanism_output = output_dir / "mechanism.yaml"
    shutil.copy2(mechanism_path, mechanism_output)
    np.savez(
        output_dir / "normalization.npz",
        state_mean=state_mean,
        state_std=state_std,
        log_dt_mean=np.asarray([log_dt_mean], dtype=np.float64),
        log_dt_std=np.asarray([log_dt_std], dtype=np.float64),
    )
    np.savez(
        output_dir / "stoichiometry.npz",
        mass_fraction_matrix=stoichiometry,
        species_names=np.asarray(species_names),
    )
    stoichiometry.tofile(output_dir / "stoichiometry.f64")
    (output_dir / "species_names.txt").write_text(
        "\n".join(species_names) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "reference_io.npz",
        physical_input=reference_input,
        current_species=reference_species.numpy(),
        model_output=reference_output,
        model_output_fields=np.asarray(
            (
                (
                    ["delta_T", *[f"delta_Y:{name}" for name in species_names]]
                    if predicts_temperature
                    else [
                        "delta_h_total",
                        *[f"delta_Y:{name}" for name in species_names],
                    ]
                )
                if emits_thermochemical_scalar
                else [f"delta_Y:{name}" for name in species_names]
            )
        ),
    )

    training_config = checkpoint.get("training_config", {})
    transform_alpha = float(training_config.get("transform_alpha", 0.1))
    scale_by_alpha = bool(
        training_config.get("transform_scale_by_alpha", True)
    )
    runtime_config = {
        "schema_version": 3 if is_neural_patankar else 1,
        "input_width": n_species + 3,
        "species_width": n_species,
        "reaction_width": runtime_reaction_width,
        "transform_alpha": transform_alpha,
        "transform_scale_by_alpha": int(scale_by_alpha),
        "output_mode": output_mode,
        "predicts_temperature": int(predicts_temperature),
        "predicts_total_enthalpy_increment": int(
            predicts_total_enthalpy_increment
        ),
        "total_enthalpy_delta_scale_j_per_kg": float(
            checkpoint.get(
                "total_enthalpy_delta_scale",
                training_config.get("total_enthalpy_delta_scale", 1.0e3),
            )
        ),
        "total_enthalpy_target_transform": training_config.get(
            "total_enthalpy_target_transform",
            TOTAL_ENTHALPY_TARGET_IDENTITY,
        ),
        "total_enthalpy_transform_alpha": float(
            training_config.get("total_enthalpy_transform_alpha", 0.1)
        ),
        "total_enthalpy_residual_mode": training_config.get(
            "total_enthalpy_residual_mode", TOTAL_ENTHALPY_RESIDUAL_NONE
        ),
        "total_enthalpy_residual_alpha": float(
            training_config.get("total_enthalpy_residual_alpha", 0.1)
        ),
        "availability_mode": (
            availability_mode if is_neural_patankar else "none"
        ),
        "availability_p_norm": (
            availability_p_norm if is_neural_patankar else 0.0
        ),
        "thermochemical_closure": effective_thermochemical_closure,
        "species_input_mode": (
            "separate_float64"
            if is_neural_patankar
            else "embedded_float32"
        ),
        "input_dtype": (
            "float64" if thermochemical_state_high_low else "float32"
        ),
        "thermochemical_state_high_low": int(
            thermochemical_state_high_low
        ),
        "thermochemical_state_low_scale": float(
            getattr(wrapper, "thermochemical_state_low_scale", float(2**24))
            if is_neural_patankar
            else float(2**24)
        ),
        "thermochemical_process_feature_mode": (
            getattr(
                wrapper,
                "thermochemical_process_feature_mode",
                THERMOCHEMICAL_PROCESS_FEATURE_NONE,
            )
            if is_neural_patankar
            else THERMOCHEMICAL_PROCESS_FEATURE_NONE
        ),
        "total_enthalpy_head_hidden_dim": (
            int(getattr(wrapper, "total_enthalpy_head_hidden_dim", 0))
            if is_neural_patankar
            else 0
        ),
        "total_enthalpy_head_depth": (
            int(getattr(wrapper, "total_enthalpy_head_depth", 0))
            if is_neural_patankar
            else 0
        ),
    }
    (output_dir / "runtime.cfg").write_text(
        "".join(f"{key}={value}\n" for key, value in runtime_config.items()),
        encoding="ascii",
    )

    manifest = {
        "schema_version": (
            4
            if predicts_total_enthalpy_increment
            else (3 if is_neural_patankar else 1)
        ),
        "artifact_type": (
            (
                (
                    "dfode-fluent-direct-delta-t-delta-y"
                    if predicts_temperature
                    else "dfode-fluent-direct-delta-h-total-delta-y"
                )
                if emits_thermochemical_scalar
                else "dfode-fluent-direct-delta-y"
            )
            if is_neural_patankar
            else "dfode-fluent-reaction-channel"
        ),
        "model_type": positive_model_type or model_type,
        "checkpoint": str(checkpoint_path),
        "mechanism": mechanism_output.name,
        "phase_name": checkpoint.get("phase_name"),
        "input": {
            "dtype": (
                "float64" if thermochemical_state_high_low else "float32"
            ),
            "shape": ["batch", n_species + 3],
            "fields": ["T", "P", *species_names, "dt"],
            "units": {
                "T": "K",
                "P": "Pa",
                "Y": "dimensionless",
                "dt": "s",
            },
        },
        "hard_layer_input": (
            {
                "dtype": "float64",
                "shape": ["batch", n_species],
                "fields": species_names,
                "meaning": (
                    "exact physical mass fractions used for "
                    "Patankar availability"
                ),
            }
            if is_neural_patankar
            else None
        ),
        "thermochemical_delta_y_features": {
            "mode": (
                getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                if is_neural_patankar
                else None
            ),
            "equation": (
                "asinh(delta_Y / species_scale)"
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else "sign(delta_Y) * abs(delta_Y) ** 0.1"
            ),
            "scale_source": (
                "training-split-only q90(abs(target_species-current_species))"
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else None
            ),
            "quantile": (
                0.9
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else None
            ),
            "floor": training_config.get(
                "thermochemical_delta_y_scale_floor", 1.0e-15
            ),
            "species_scales": (
                wrapper.thermochemical_delta_y_scales.detach().cpu().tolist()
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else None
            ),
            "consumers": (
                [
                    *(
                        ["auxiliary-temperature"]
                        if getattr(
                            wrapper,
                            "thermochemical_aux_temperature",
                            False,
                        )
                        else []
                    ),
                    "total-enthalpy",
                ]
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_delta_y_feature_mode",
                    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
                )
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else []
            ),
            "physical_delta_y_transformed": False,
            "species_reaction_encoder_uses_feature": False,
        },
        "thermochemical_process_features": {
            "mode": (
                getattr(
                    wrapper,
                    "thermochemical_process_feature_mode",
                    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                )
                if is_neural_patankar
                else THERMOCHEMICAL_PROCESS_FEATURE_NONE
            ),
            "physical_equation": (
                "G = process_extent @ (2 * consumption + "
                "process_stoich).T"
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_process_feature_mode",
                    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                )
                == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
                else (
                    "process_extent = demand * process_availability"
                    if is_neural_patankar
                    and getattr(
                        wrapper,
                        "thermochemical_process_feature_mode",
                        THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                    )
                    == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
                    else None
                )
            ),
            "feature_equation": (
                "G ** 0.1"
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_process_feature_mode",
                    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                )
                == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
                else (
                    "where(process_extent == 0, 0, "
                    "clamp_min(process_extent, 1e-30) ** 0.1)"
                    if is_neural_patankar
                    and getattr(
                        wrapper,
                        "thermochemical_process_feature_mode",
                        THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                    )
                    == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
                    else None
                )
            ),
            "physical_dtype": "float64",
            "feature_dtype": "model hidden dtype",
            "consumers": (
                ["total-enthalpy"]
                if is_neural_patankar
                and getattr(
                    wrapper,
                    "thermochemical_process_feature_mode",
                    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                )
                != THERMOCHEMICAL_PROCESS_FEATURE_NONE
                else []
            ),
            "auxiliary_temperature_uses_feature": False,
            "species_reaction_encoder_uses_feature": False,
        },
        "total_enthalpy_head": {
            "hidden_dim_override": training_config.get(
                "total_enthalpy_head_hidden_dim", 0
            ),
            "depth_override": training_config.get(
                "total_enthalpy_head_depth", 0
            ),
            "resolved_hidden_dim": (
                int(getattr(wrapper, "total_enthalpy_head_hidden_dim", 0))
                if is_neural_patankar
                else None
            ),
            "resolved_depth": (
                int(getattr(wrapper, "total_enthalpy_head_depth", 0))
                if is_neural_patankar
                else None
            ),
            "auxiliary_temperature_head_uses_overrides": False,
            "residual_mode": training_config.get(
                "total_enthalpy_residual_mode",
                TOTAL_ENTHALPY_RESIDUAL_NONE,
            ),
            "residual_alpha": float(
                training_config.get("total_enthalpy_residual_alpha", 0.1)
            ),
            "baseline_frozen": (
                training_config.get(
                    "total_enthalpy_residual_mode",
                    TOTAL_ENTHALPY_RESIDUAL_NONE,
                )
                != TOTAL_ENTHALPY_RESIDUAL_NONE
            ),
            "physical_equation": (
                "delta_h = scale * (baseline_scaled + "
                "inverse_signed_power(residual_head, residual_alpha))"
                if training_config.get(
                    "total_enthalpy_residual_mode",
                    TOTAL_ENTHALPY_RESIDUAL_NONE,
                )
                != TOTAL_ENTHALPY_RESIDUAL_NONE
                else None
            ),
        },
        "thermochemical_state_features": {
            "mode": (
                "float32-high-scaled-float32-low"
                if thermochemical_state_high_low
                else "float32-high-only"
            ),
            "space": "normalized-state",
            "low_equation": (
                "float32((x64 - float64(float32(x64))) * low_scale)"
                if thermochemical_state_high_low
                else None
            ),
            "low_scale": float(
                getattr(
                    wrapper,
                    "thermochemical_state_low_scale",
                    float(2**24),
                )
                if is_neural_patankar
                else float(2**24)
            ),
            "consumers": (
                ["auxiliary-temperature", "total-enthalpy"]
                if thermochemical_state_high_low
                else []
            ),
            "species_reaction_encoder_uses_low": False,
        },
        "output": {
            "dtype": "float64" if is_neural_patankar else "float32",
            "shape": [
                "batch",
                runtime_reaction_width,
            ],
            "fields": (
                (
                    ["delta_T", *[f"delta_Y:{name}" for name in species_names]]
                    if predicts_temperature
                    else [
                        "delta_h_total",
                        *[f"delta_Y:{name}" for name in species_names],
                    ]
                )
                if emits_thermochemical_scalar
                else (
                    [f"delta_Y:{name}" for name in species_names]
                    if is_neural_patankar
                    else [f"reaction_channel:{index}" for index in range(n_reactions)]
                )
            ),
            "meaning": (
                (
                    "independently predicted delta_T followed by positive "
                    "stoichiometrically conservative delta_Y"
                    if predicts_temperature
                    else (
                        "predicted Fluent total-enthalpy increment followed by "
                        "positive stoichiometrically conservative delta_Y"
                        if predicts_total_enthalpy_increment
                        else (
                        "positive stoichiometrically conservative delta_Y "
                        "only; no temperature increment is emitted"
                        )
                    )
                )
                if is_neural_patankar
                else "signed-power-transformed integrated reaction extents"
            ),
        },
        "thermochemical_closure": (
            {
                "mode": PREDICTED_DELTA_T_CLOSURE,
                "predicts_temperature": True,
                "runtime_contract": "artifact_info.predicts_temperature=true",
                "temperature_update": "T_next = T_current + delta_T",
                "temperature_owner": "dfode-model",
                "enthalpy_projection": False,
            }
            if predicts_temperature
            else (
                {
                    "mode": FLUENT_TOTAL_ENTHALPY_INCREMENT_CLOSURE,
                    "predicts_temperature": False,
                    "predicts_total_enthalpy_increment": True,
                    "runtime_contract": (
                        "artifact_info.predicts_temperature=false; "
                        "artifact_info.predicts_total_enthalpy_increment=true"
                    ),
                    "temperature_owner": "ansys-fluent",
                    "enthalpy_increment_field": "delta_h_total",
                    "enthalpy_increment_units": "J/kg",
                    "target_contract": checkpoint.get(
                        "thermochemical_target_contract",
                        {
                            "mode": DELTA_H_TOTAL_MODE,
                            "label_backend": "fluent-native-di",
                            "before_dataset": "pairs/h_total_before",
                            "after_dataset": "pairs/h_total_after",
                            "target_equation": (
                                "delta_h_total = h_total_after - h_total_before"
                            ),
                            "target_units": "J/kg",
                            "total_enthalpy_delta_scale_j_per_kg": float(
                                training_config.get(
                                    "total_enthalpy_delta_scale", 1.0e3
                                )
                            ),
                        },
                    ),
                    "operation_order": [
                        "evaluate h_total_before with Fluent thermodynamics",
                        "apply Y_next = Y_current + delta_Y",
                        "set h_total_next = h_total_before + delta_h_total",
                        "recover T_next with Fluent reference enthalpy",
                    ],
                    "temperature_recovery": (
                        "T_next = Temperature(h_total_before + delta_h_total - "
                        "Reference_Enthalpy(Y_next), Y_next)"
                    ),
                    "thermodynamics_backend": "ansys-fluent",
                    "python_thermodynamics_equivalence_claim": "none",
                }
                if predicts_total_enthalpy_increment
                else {
                "mode": FLUENT_CONSTANT_ENTHALPY_CLOSURE,
                "predicts_temperature": False,
                "runtime_contract": "artifact_info.predicts_temperature=false",
                "temperature_owner": "ansys-fluent",
                "operation_order": [
                    "preserve pre-chemistry Fluent mixture enthalpy",
                    "apply Y_next = Y_current + delta_Y",
                    "recover T_next with Fluent thermodynamics",
                ],
                "closure_equation": (
                    "h_Fluent(T_next, p, Y_next) = "
                    "h_Fluent(T_current, p, Y_current)"
                ),
                "thermodynamics_backend": "ansys-fluent",
                "python_thermodynamics_equivalence_claim": "none",
                }
            )
        ),
        "hard_layer": {
            "equation": (
                "delta_Y = PatankarResourceAllocation(Y, demand)"
                if is_neural_patankar
                else "delta_Y = (W*S) @ reaction_extent"
            ),
            "dtype": "float64",
            "current_species_dtype": (
                "float64" if is_neural_patankar else None
            ),
            "current_species_source": (
                "separate_input" if is_neural_patankar else None
            ),
            "transform_alpha": transform_alpha,
            "transform_scale_by_alpha": scale_by_alpha,
            "limiter_implementation": (
                (
                    "sparse-reactant-index-smooth-safe"
                    if availability_mode == "smooth-safe"
                    else "sparse-reactant-index-hard"
                )
                if is_neural_patankar
                else None
            ),
            "availability_mode": (
                availability_mode if is_neural_patankar else None
            ),
            "availability_p_norm": (
                availability_p_norm if is_neural_patankar else None
            ),
        },
        "species_names": species_names,
        "n_reactions": n_reactions,
        "checksums": {
            "model_sha256": _sha256(model_path),
            "mechanism_sha256": _sha256(mechanism_output),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "reference_max_abs_error": max_abs_error,
        "sparse_equivalence_max_abs_error": (
            sparse_equivalence_max_abs_error
        ),
        "training_config": training_config,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a DFODE interval checkpoint for native Fluent inference."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--thermochemical-closure",
        choices=THERMOCHEMICAL_CLOSURES,
        default=None,
        help=(
            "Override the checkpoint thermochemical contract. By default, "
            "legacy checkpoints emit delta_T and delta-h-total checkpoints "
            "emit delta_h_total. Species-only constant-enthalpy export remains "
            "available for controlled comparisons."
        ),
    )
    args = parser.parse_args()
    manifest = export_artifact(
        args.checkpoint.resolve(),
        args.mechanism.resolve(),
        args.output.resolve(),
        thermochemical_closure=args.thermochemical_closure,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
