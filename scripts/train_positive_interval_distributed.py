#!/usr/bin/env python3
"""Deterministic distributed training for positive interval models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import cantera as ct
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel

from dfode_kit.data.fluent_sensitivity import (
    FLUENT_NATIVE_DI_BACKEND,
    deterministic_paired_batch_indices,
)
from dfode_kit.data.interval_pairs import load_interval_pair_arrays
from dfode_kit.models.positive_interval import (
    DELTA_H_TOTAL_MODE,
    DELTA_TEMPERATURE_MODE,
    TOTAL_ENTHALPY_TARGET_IDENTITY,
    TOTAL_ENTHALPY_TARGET_SIGNED_POWER,
    TOTAL_ENTHALPY_TARGET_TRANSFORMS,
    TOTAL_ENTHALPY_RESIDUAL_MODES,
    TOTAL_ENTHALPY_RESIDUAL_NONE,
    TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER,
    THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH,
    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
    THERMOCHEMICAL_DELTA_Y_FEATURE_MODES,
    THERMOCHEMICAL_HEAD_INPUT_LEGACY,
    THERMOCHEMICAL_HEAD_INPUT_MODES,
    THERMOCHEMICAL_OUTPUT_MODES,
    THERMOCHEMICAL_PROCESS_FEATURE_MODES,
    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
    THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS,
    THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER,
)
from dfode_kit.physics.atom_conservation import reaction_stoichiometry
from dfode_kit.training.positive_interval import (
    PositiveIntervalTrainingConfig,
    _build_model,
    compute_total_enthalpy_sign_statistics,
    compute_thermochemical_delta_y_scales,
    _deployment_loss_config,
    _deployment_loss_for_batch,
    _evaluate_jvp_partition,
    _jvp_loss_config,
    _load_total_enthalpy_increment_labels,
    _prepare_jvp_training_data,
    _require_deployment_label_sources,
    _signed_power,
    _validate_row_tail_configuration,
    _validate_thermochemical_configuration,
    total_enthalpy_sign_balanced_loss,
)
from dfode_kit.training.fluent_deployment_losses import (
    fluent_di_jvp_consistency_losses,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-output", default=None)
    parser.add_argument("--resume-output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--initial-checkpoint", default=None)
    parser.add_argument(
        "--normalization-source",
        choices=("train", "initial-checkpoint"),
        default="train",
        help=(
            "Compute input normalizers from the training source (default), or "
            "preserve them from --initial-checkpoint during continuation"
        ),
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size-per-device", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--tp-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--thermochemical-output-mode",
        choices=THERMOCHEMICAL_OUTPUT_MODES,
        default=DELTA_TEMPERATURE_MODE,
    )
    parser.add_argument("--temperature-delta-scale", type=float, default=10.0)
    parser.add_argument(
        "--total-enthalpy-delta-scale", type=float, default=1.0e3
    )
    parser.add_argument(
        "--total-enthalpy-delta-loss-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--total-enthalpy-target-transform",
        choices=TOTAL_ENTHALPY_TARGET_TRANSFORMS,
        default=TOTAL_ENTHALPY_TARGET_IDENTITY,
    )
    parser.add_argument(
        "--total-enthalpy-transform-alpha", type=float, default=0.1
    )
    parser.add_argument(
        "--total-enthalpy-transformed-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--total-enthalpy-residual-mode",
        choices=TOTAL_ENTHALPY_RESIDUAL_MODES,
        default=TOTAL_ENTHALPY_RESIDUAL_NONE,
    )
    parser.add_argument(
        "--total-enthalpy-residual-alpha", type=float, default=0.1
    )
    parser.add_argument(
        "--total-enthalpy-residual-transformed-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--total-enthalpy-sign-balanced-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for auxiliary equal negative/positive stratum NMAE; "
            "the existing raw-MAE enthalpy objective remains primary"
        ),
    )
    parser.add_argument(
        "--thermochemical-head-input-mode",
        choices=THERMOCHEMICAL_HEAD_INPUT_MODES,
        default=THERMOCHEMICAL_HEAD_INPUT_LEGACY,
    )
    parser.add_argument("--thermochemical-head-hidden-dim", type=int, default=0)
    parser.add_argument("--thermochemical-head-depth", type=int, default=1)
    parser.add_argument(
        "--total-enthalpy-head-hidden-dim", type=int, default=0
    )
    parser.add_argument("--total-enthalpy-head-depth", type=int, default=0)
    parser.add_argument(
        "--thermochemical-aux-temperature", action="store_true"
    )
    parser.add_argument(
        "--thermochemical-aux-temperature-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--thermochemical-delta-y-feature-mode",
        choices=THERMOCHEMICAL_DELTA_Y_FEATURE_MODES,
        default=THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
    )
    parser.add_argument(
        "--thermochemical-delta-y-scale-floor",
        type=float,
        default=1.0e-15,
    )
    parser.add_argument(
        "--thermochemical-process-feature-mode",
        choices=THERMOCHEMICAL_PROCESS_FEATURE_MODES,
        default=THERMOCHEMICAL_PROCESS_FEATURE_NONE,
    )
    parser.add_argument(
        "--thermochemical-state-high-low", action="store_true"
    )
    parser.add_argument(
        "--thermochemical-state-low-scale",
        type=float,
        default=float(2**24),
    )
    parser.add_argument("--temperature-selection-weight", type=float, default=0.0)
    parser.add_argument("--temperature-rmse-weight", type=float, default=0.0)
    parser.add_argument(
        "--temperature-rmse-selection-weight", type=float, default=0.0
    )
    parser.add_argument("--temperature-head-only", action="store_true")
    parser.add_argument("--thermochemical-head-only", action="store_true")
    parser.add_argument("--total-enthalpy-head-only", action="store_true")
    parser.add_argument(
        "--total-enthalpy-residual-head-only", action="store_true"
    )
    parser.add_argument("--reset-thermochemical-head", action="store_true")
    parser.add_argument("--reset-total-enthalpy-head", action="store_true")
    parser.add_argument("--state-loss-weight", type=float, default=0.0)
    parser.add_argument("--delta-loss-weight", type=float, default=1.0)
    parser.add_argument("--relative-delta-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--row-tail-relative-delta-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--row-tail-relative-delta-fraction", type=float, default=0.1
    )
    parser.add_argument(
        "--row-tail-relative-delta-floor", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--row-tail-relative-delta-selection-weight", type=float, default=0.0
    )
    parser.add_argument("--source-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-loss-scale", type=float, default=1.0)
    parser.add_argument(
        "--mixture-molecular-weight-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--mixture-molecular-weight-loss-scale", type=float, default=1.0
    )
    parser.add_argument("--density-increment-loss-weight", type=float, default=0.0)
    parser.add_argument("--density-increment-loss-scale", type=float, default=1.0)
    parser.add_argument("--formation-energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--direction-loss-weight", type=float, default=0.25)
    parser.add_argument("--opposing-extent-loss-weight", type=float, default=0.0)
    parser.add_argument("--jvp-dataset", default=None)
    parser.add_argument("--source-jvp-loss-weight", type=float, default=0.0)
    parser.add_argument("--source-jvp-loss-scale", type=float, default=1.0)
    parser.add_argument("--extent-scale", type=float, default=1.0e-4)
    parser.add_argument("--demand-bias-init", type=float, default=-8.0)
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--non-deterministic", action="store_true")
    return parser.parse_args(argv)


def _training_config_from_args(
    args: argparse.Namespace, *, world_size: int
) -> PositiveIntervalTrainingConfig:
    return PositiveIntervalTrainingConfig(
        variant="neural-patankar",
        epochs=args.epochs,
        batch_size=args.batch_size_per_device * world_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        transform_alpha=args.transform_alpha,
        tp_loss_weight=args.tp_loss_weight,
        state_loss_weight=args.state_loss_weight,
        delta_loss_weight=args.delta_loss_weight,
        relative_delta_loss_weight=args.relative_delta_loss_weight,
        row_tail_relative_delta_loss_weight=(
            args.row_tail_relative_delta_loss_weight
        ),
        row_tail_relative_delta_fraction=(
            args.row_tail_relative_delta_fraction
        ),
        row_tail_relative_delta_floor=args.row_tail_relative_delta_floor,
        row_tail_relative_delta_selection_weight=(
            args.row_tail_relative_delta_selection_weight
        ),
        source_loss_weight=args.source_loss_weight,
        source_loss_scale=args.source_loss_scale,
        mixture_molecular_weight_loss_weight=(
            args.mixture_molecular_weight_loss_weight
        ),
        mixture_molecular_weight_loss_scale=(
            args.mixture_molecular_weight_loss_scale
        ),
        density_increment_loss_weight=args.density_increment_loss_weight,
        density_increment_loss_scale=args.density_increment_loss_scale,
        formation_energy_loss_weight=args.formation_energy_loss_weight,
        jvp_dataset=args.jvp_dataset,
        source_jvp_loss_weight=args.source_jvp_loss_weight,
        source_jvp_loss_scale=args.source_jvp_loss_scale,
        thermochemical_output_mode=args.thermochemical_output_mode,
        thermochemical_head_input_mode=args.thermochemical_head_input_mode,
        thermochemical_head_hidden_dim=args.thermochemical_head_hidden_dim,
        thermochemical_head_depth=args.thermochemical_head_depth,
        total_enthalpy_head_hidden_dim=(
            args.total_enthalpy_head_hidden_dim
        ),
        total_enthalpy_head_depth=args.total_enthalpy_head_depth,
        thermochemical_aux_temperature=(
            args.thermochemical_aux_temperature
        ),
        thermochemical_aux_temperature_loss_weight=(
            args.thermochemical_aux_temperature_loss_weight
        ),
        thermochemical_delta_y_feature_mode=(
            args.thermochemical_delta_y_feature_mode
        ),
        thermochemical_delta_y_scale_floor=(
            args.thermochemical_delta_y_scale_floor
        ),
        thermochemical_process_feature_mode=(
            args.thermochemical_process_feature_mode
        ),
        thermochemical_state_high_low=args.thermochemical_state_high_low,
        thermochemical_state_low_scale=(
            args.thermochemical_state_low_scale
        ),
        temperature_delta_scale=args.temperature_delta_scale,
        total_enthalpy_delta_scale=args.total_enthalpy_delta_scale,
        total_enthalpy_delta_loss_weight=(
            args.total_enthalpy_delta_loss_weight
        ),
        total_enthalpy_target_transform=(
            args.total_enthalpy_target_transform
        ),
        total_enthalpy_transform_alpha=(
            args.total_enthalpy_transform_alpha
        ),
        total_enthalpy_transformed_loss_weight=(
            args.total_enthalpy_transformed_loss_weight
        ),
        total_enthalpy_residual_mode=args.total_enthalpy_residual_mode,
        total_enthalpy_residual_alpha=args.total_enthalpy_residual_alpha,
        total_enthalpy_residual_transformed_loss_weight=(
            args.total_enthalpy_residual_transformed_loss_weight
        ),
        total_enthalpy_sign_balanced_loss_weight=(
            args.total_enthalpy_sign_balanced_loss_weight
        ),
        temperature_rmse_weight=args.temperature_rmse_weight,
        temperature_rmse_selection_weight=(
            args.temperature_rmse_selection_weight
        ),
        direction_loss_weight=args.direction_loss_weight,
        opposing_extent_loss_weight=args.opposing_extent_loss_weight,
        extent_scale=args.extent_scale,
        demand_bias_init=args.demand_bias_init,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        log_every=args.log_every,
    )


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _validate_normalization_arguments(args: argparse.Namespace) -> None:
    if (
        args.normalization_source == "initial-checkpoint"
        and not args.initial_checkpoint
    ):
        raise ValueError(
            "normalization-source=initial-checkpoint requires "
            "--initial-checkpoint"
        )


def _normalizers_from_initial_checkpoint(
    checkpoint: dict,
    *,
    state_dimension: int,
    checkpoint_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected_shapes = {
        "state_mean": (state_dimension,),
        "state_std": (state_dimension,),
        "log_dt_mean": (1,),
        "log_dt_std": (1,),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, expected_shape in expected_shapes.items():
        if name not in checkpoint:
            raise ValueError(
                f"Initial checkpoint {checkpoint_path!r} is missing required "
                f"normalizer {name!r}"
            )
        value = checkpoint[name]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Initial checkpoint normalizer {name!r} cannot be converted "
                "to float64"
            ) from exc
        if array.shape != expected_shape:
            raise ValueError(
                f"Initial checkpoint normalizer {name!r} has shape "
                f"{array.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"Initial checkpoint normalizer {name!r} contains non-finite "
                "values"
            )
        if name.endswith("_std") and not np.all(array > 0.0):
            raise ValueError(
                f"Initial checkpoint normalizer {name!r} must be strictly "
                "positive"
            )
        arrays[name] = np.array(array, dtype=np.float64, copy=True)
    return (
        arrays["state_mean"],
        arrays["state_std"],
        arrays["log_dt_mean"],
        arrays["log_dt_std"],
    )


def _decode_phase(attrs: dict, override: str | None) -> str | None:
    if override:
        return override
    value = attrs.get("phase_name")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value) if value else None


def _load_core(path: str):
    loaded = load_interval_pair_arrays(path, dtype=np.float64)
    return loaded[0], loaded[1], loaded[2], loaded[5], loaded[6]


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _normalize_inputs(
    current: np.ndarray,
    dt: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: np.ndarray,
    log_dt_std: np.ndarray,
    *,
    high_low: bool,
    low_scale: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    normalized_float64 = (current - state_mean) / state_std
    x = normalized_float64.astype(np.float32)
    x_low = (
        (
            (normalized_float64 - x.astype(np.float64)) * low_scale
        ).astype(np.float32)
        if high_low
        else None
    )
    log_dt = np.log(np.maximum(dt, 1.0e-300))[:, None]
    t = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)
    return x, x_low, t


def _row_relative_delta_errors(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    numerator = torch.sum(torch.abs(predicted_delta - target_delta), dim=1)
    denominator = torch.sum(torch.abs(target_delta), dim=1).clamp_min(
        denominator_floor
    )
    return numerator / denominator


def _row_tail_relative_delta_loss(
    predicted_delta: torch.Tensor,
    target_delta: torch.Tensor,
    *,
    fraction: float,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_errors = _row_relative_delta_errors(
        predicted_delta,
        target_delta,
        denominator_floor,
    )
    tail_count = min(
        row_errors.numel(),
        max(1, int(np.ceil(fraction * row_errors.numel()))),
    )
    tail = torch.topk(row_errors, tail_count, sorted=False).values
    return torch.mean(torch.log1p(tail)), row_errors


def _batch_loss(
    model,
    x: torch.Tensor,
    thermochemical_state_low: torch.Tensor | None,
    t: torch.Tensor,
    y0: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    alpha: float,
    tp_weight: float,
    state_weight: float,
    delta_weight: float,
    relative_delta_weight: float,
    row_tail_relative_delta_weight: float,
    row_tail_relative_delta_fraction: float,
    row_tail_relative_delta_floor: float,
    formation_enthalpy: torch.Tensor,
    formation_energy_weight: float,
    direction_weight: float,
    opposing_extent_weight: float,
    temperature_delta_scale: float,
    temperature_rmse_weight: float,
    thermochemical_output_mode: str,
    total_enthalpy_delta_scale: float,
    total_enthalpy_delta_loss_weight: float,
    total_enthalpy_target_transform: str,
    total_enthalpy_transform_alpha: float,
    total_enthalpy_transformed_loss_weight: float,
    total_enthalpy_residual_mode: str,
    total_enthalpy_residual_alpha: float,
    total_enthalpy_residual_transformed_loss_weight: float,
    total_enthalpy_sign_balanced_loss_weight: float,
    total_enthalpy_negative_abs_target_sum: float | None,
    total_enthalpy_positive_abs_target_sum: float | None,
    total_enthalpy_training_sample_count: int,
    thermochemical_aux_temperature_loss_weight: float,
    thermochemical_state_low_scale: float,
    target_delta_h_total: torch.Tensor | None,
    extent_scale: float,
    dt: torch.Tensor,
    molecular_weights: torch.Tensor,
    zero_species_enthalpy: torch.Tensor,
    deployment_loss_config,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
]:
    output = model(
        x,
        t,
        current_species=y0,
        thermochemical_state_low=thermochemical_state_low,
    )
    prediction = output["next_state"].clone()
    thermochemical_normalized_state = x.to(torch.float64)
    if thermochemical_state_low is not None:
        thermochemical_normalized_state = (
            thermochemical_normalized_state
            + thermochemical_state_low.to(torch.float64)
            / thermochemical_state_low_scale
        )
    current_temperature = (
        thermochemical_normalized_state[:, 0] * state_std[0] + state_mean[0]
    )
    zero = output["delta_species"].sum() * 0.0
    tp_loss = zero
    total_enthalpy_delta_loss = zero
    transformed_total_enthalpy_loss = zero
    residual_transformed_total_enthalpy_loss = zero
    sign_balanced_total_enthalpy_loss = zero
    auxiliary_temperature_loss = zero
    if thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
        target_temperature_delta = target[:, 0] - current_temperature
        thermochemical_prediction = output["delta_temperature"].reshape(-1)
        normalized_temperature_error = (
            thermochemical_prediction - target_temperature_delta
        ) / temperature_delta_scale
        tp_loss = (
            torch.mean(torch.abs(normalized_temperature_error))
            + temperature_rmse_weight
            * torch.sqrt(torch.mean(normalized_temperature_error.square()))
        )
        prediction[:, 0] = (
            current_temperature + thermochemical_prediction - state_mean[0]
        ) / state_std[0]
    else:
        if target_delta_h_total is None:
            raise ValueError(
                "delta-h-total mode requires total-enthalpy increment labels"
            )
        thermochemical_prediction = output["delta_h_total"].reshape(-1)
        total_enthalpy_delta_loss = torch.mean(
            torch.abs(
                thermochemical_prediction - target_delta_h_total.reshape(-1)
            )
            / total_enthalpy_delta_scale
        )
        if (
            total_enthalpy_target_transform
            == TOTAL_ENTHALPY_TARGET_SIGNED_POWER
        ):
            transformed_total_enthalpy_loss = functional.l1_loss(
                output["delta_h_total_head_space"]
                .reshape(-1)
                .to(torch.float64),
                _signed_power(
                    target_delta_h_total.reshape(-1)
                    / total_enthalpy_delta_scale,
                    total_enthalpy_transform_alpha,
                ),
            )
        if (
            total_enthalpy_residual_mode
            == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
        ):
            baseline_scaled = output[
                "delta_h_total_baseline_scaled"
            ].reshape(-1).detach()
            residual_target_scaled = (
                target_delta_h_total.reshape(-1)
                / total_enthalpy_delta_scale
                - baseline_scaled
            )
            residual_transformed_total_enthalpy_loss = functional.l1_loss(
                output["delta_h_total_residual_head_space"]
                .reshape(-1)
                .to(torch.float64),
                _signed_power(
                    residual_target_scaled,
                    total_enthalpy_residual_alpha,
                ),
            )
        if total_enthalpy_sign_balanced_loss_weight > 0.0:
            sign_balanced_total_enthalpy_loss = (
                total_enthalpy_sign_balanced_loss(
                    thermochemical_prediction,
                    target_delta_h_total,
                    negative_abs_target_sum=float(
                        total_enthalpy_negative_abs_target_sum
                    ),
                    positive_abs_target_sum=float(
                        total_enthalpy_positive_abs_target_sum
                    ),
                    training_sample_count=(
                        total_enthalpy_training_sample_count
                    ),
                )
            )
        if "aux_delta_temperature" in output:
            target_temperature_delta = target[:, 0] - current_temperature
            auxiliary_temperature_loss = torch.mean(
                torch.abs(
                    output["aux_delta_temperature"].reshape(-1)
                    - target_temperature_delta
                )
                / temperature_delta_scale
            )
    predicted_y = prediction[:, 2:]
    target_y = target[:, 2:]
    y_loss = functional.l1_loss(
        _signed_power(predicted_y, alpha),
        _signed_power(target_y, alpha),
    )
    delta_loss = functional.l1_loss(
        _signed_power(predicted_y - y0, alpha),
        _signed_power(target_y - y0, alpha),
    )
    predicted_delta = predicted_y - y0
    target_delta = target_y - y0
    predicted_temperature = (
        current_temperature + thermochemical_prediction
        if thermochemical_output_mode == DELTA_TEMPERATURE_MODE
        else current_temperature
    )
    current_pressure = (
        thermochemical_normalized_state[:, 1] * state_std[1] + state_mean[1]
    )
    deployment = _deployment_loss_for_batch(
        loss_config=deployment_loss_config,
        current_y=y0,
        predicted_delta_y=predicted_delta,
        target_delta_y=target_delta,
        current_temperature=current_temperature,
        predicted_temperature=predicted_temperature,
        target_temperature=target[:, 0],
        pressure=current_pressure,
        dt=dt,
        molecular_weights=molecular_weights,
        zero_species_enthalpy=zero_species_enthalpy,
    )
    deployment_terms = deployment["terms"]
    relative_delta_loss = torch.sum(torch.abs(predicted_delta - target_delta)) / torch.sum(
        torch.abs(target_delta)
    ).clamp_min(torch.finfo(target_delta.dtype).tiny)
    row_tail_relative_delta_loss, _ = _row_tail_relative_delta_loss(
        predicted_delta,
        target_delta,
        fraction=row_tail_relative_delta_fraction,
        denominator_floor=row_tail_relative_delta_floor,
    )
    predicted_energy = -torch.sum(predicted_delta * formation_enthalpy[None, :], dim=1)
    target_energy = -torch.sum(target_delta * formation_enthalpy[None, :], dim=1)
    formation_energy_loss = torch.sum(torch.abs(predicted_energy - target_energy)) / torch.sum(
        torch.abs(target_energy)
    ).clamp_min(torch.finfo(target_energy.dtype).tiny)
    predicted_transformed = _signed_power(predicted_delta, alpha)
    target_transformed = _signed_power(target_delta, alpha)
    target_norm = torch.linalg.vector_norm(target_transformed, dim=1)
    reactive = target_norm > 1.0e-12
    if torch.any(reactive):
        cosine = functional.cosine_similarity(
            predicted_transformed[reactive],
            target_transformed[reactive],
            dim=1,
            eps=1.0e-30,
        )
        direction_loss = 1.0 - cosine.mean()
    else:
        direction_loss = predicted_delta.sum() * 0.0
    opposing_extent_loss = torch.mean(
        torch.minimum(output["forward_extent"], output["reverse_extent"])
    ) / extent_scale
    total_loss = (
        tp_weight * tp_loss
        + state_weight * y_loss
        + delta_weight * delta_loss
        + relative_delta_weight * relative_delta_loss
        + row_tail_relative_delta_weight * row_tail_relative_delta_loss
        + formation_energy_weight * formation_energy_loss
        + direction_weight * direction_loss
        + opposing_extent_weight * opposing_extent_loss
        + total_enthalpy_delta_loss_weight * total_enthalpy_delta_loss
        + total_enthalpy_transformed_loss_weight
        * transformed_total_enthalpy_loss
        + total_enthalpy_residual_transformed_loss_weight
        * residual_transformed_total_enthalpy_loss
        + thermochemical_aux_temperature_loss_weight
        * auxiliary_temperature_loss
        + deployment["total"]
    )
    if total_enthalpy_sign_balanced_loss_weight > 0.0:
        total_loss = total_loss + (
            total_enthalpy_sign_balanced_loss_weight
            * sign_balanced_total_enthalpy_loss
        )
    return (
        total_loss,
        (
            tp_loss,
            y_loss,
            delta_loss,
            relative_delta_loss,
            row_tail_relative_delta_loss,
            formation_energy_loss,
            direction_loss,
            opposing_extent_loss,
            total_enthalpy_delta_loss,
            auxiliary_temperature_loss,
            deployment_terms["source"]["scaled"],
            deployment_terms["mixture_molecular_weight"]["scaled"],
            deployment_terms["density_increment"]["scaled"],
        ),
        prediction,
        thermochemical_prediction,
    )


def _distributed_jvp_loss_for_indices(
    model,
    jvp_data,
    indices: np.ndarray,
    *,
    config: PositiveIntervalTrainingConfig,
    loss_config,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    log_dt_mean: torch.Tensor,
    log_dt_std: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    selected = np.asarray(indices, dtype=np.int64)
    batch = jvp_data.batch
    base = torch.from_numpy(batch.x[selected]).to(device=device)
    perturbed = torch.from_numpy(batch.x_perturbed[selected]).to(device=device)
    physical_state = torch.cat([base, perturbed], dim=0).to(torch.float64)
    physical_dt = torch.from_numpy(batch.dt[selected]).to(
        device=device, dtype=torch.float64
    )
    paired_dt = torch.cat([physical_dt, physical_dt], dim=0)
    normalized_state_float64 = (
        (physical_state - state_mean[None, :]) / state_std[None, :]
    )
    normalized_state = normalized_state_float64.to(torch.float32)
    thermochemical_state_low = None
    if config.thermochemical_state_high_low:
        thermochemical_state_low = (
            (
                normalized_state_float64
                - normalized_state.to(torch.float64)
            )
            * config.thermochemical_state_low_scale
        ).to(torch.float32)
    normalized_log_dt = (
        (torch.log(paired_dt)[:, None] - log_dt_mean[None, :])
        / log_dt_std[None, :]
    ).to(torch.float32)
    output = model(
        normalized_state,
        normalized_log_dt,
        current_species=physical_state[:, 2:],
        thermochemical_state_low=thermochemical_state_low,
    )
    predicted_endpoint = torch.cat(
        [physical_state[:, :2], output["next_species"]], dim=1
    )
    count = selected.size
    epsilon = torch.from_numpy(batch.epsilon[selected]).to(
        device=device, dtype=torch.float64
    )
    return fluent_di_jvp_consistency_losses(
        current_state=base,
        perturbed_state=perturbed,
        predicted_endpoint=predicted_endpoint[:count],
        predicted_perturbed_endpoint=predicted_endpoint[count:],
        target_di_endpoint=torch.from_numpy(
            batch.phi_di_x[selected]
        ).to(device=device, dtype=torch.float64),
        target_di_perturbed_endpoint=torch.from_numpy(
            batch.phi_di_x_perturbed[selected]
        ).to(device=device, dtype=torch.float64),
        epsilon=epsilon,
        dt=physical_dt,
        config=loss_config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )


def _make_checkpoint(
    *,
    model,
    config: PositiveIntervalTrainingConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: np.ndarray,
    log_dt_std: np.ndarray,
    gas,
    train_source: str,
    validation_source: str,
    mechanism: str,
    history: list[dict],
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    world_size: int,
    batch_size_per_device: int,
    learning_rate: float,
    minimum_learning_rate: float,
    normalization_source: str,
    normalization_provenance: dict[str, object],
) -> dict:
    stoich = reaction_stoichiometry(gas)
    return {
        "positive_model_type": config.variant,
        "net": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "log_dt_mean": log_dt_mean,
        "log_dt_std": log_dt_std,
        "normalization_source": normalization_source,
        "normalization_provenance": dict(normalization_provenance),
        "stoichiometric_matrix": stoich.net,
        "molecular_weights": np.asarray(
            gas.molecular_weights, dtype=np.float64
        ),
        "species_names": list(gas.species_names),
        "phase_name": gas.name,
        "source_path": str(train_source),
        "validation_path": str(validation_source),
          "mechanism": str(mechanism),
          "thermochemical_output_mode": config.thermochemical_output_mode,
          "total_enthalpy_head_hidden_dim": (
              config.total_enthalpy_head_hidden_dim
          ),
          "total_enthalpy_head_depth": config.total_enthalpy_head_depth,
          "thermochemical_process_feature_mode": (
              config.thermochemical_process_feature_mode
          ),
          "total_enthalpy_delta_scale": config.total_enthalpy_delta_scale,
          "total_enthalpy_residual_contract": {
              "mode": config.total_enthalpy_residual_mode,
              "baseline_head": "total_enthalpy_delta_head",
              "baseline_transform": TOTAL_ENTHALPY_TARGET_IDENTITY,
              "baseline_frozen": (
                  config.total_enthalpy_residual_mode
                  != TOTAL_ENTHALPY_RESIDUAL_NONE
              ),
              "baseline_checkpoint": normalization_provenance.get(
                  "initial_checkpoint"
              ),
              "residual_head": (
                  "total_enthalpy_residual_head"
                  if config.total_enthalpy_residual_mode
                  != TOTAL_ENTHALPY_RESIDUAL_NONE
                  else None
              ),
              "residual_alpha": config.total_enthalpy_residual_alpha,
              "transformed_loss_weight": (
                  config.total_enthalpy_residual_transformed_loss_weight
              ),
              "physical_equation": (
                  "delta_h = scale * (baseline_scaled + "
                  "inverse_signed_power(residual_head, residual_alpha))"
                  if config.total_enthalpy_residual_mode
                  != TOTAL_ENTHALPY_RESIDUAL_NONE
                  else None
              ),
          },
          "thermochemical_state_feature_contract": {
              "mode": (
                  "float32-high-scaled-float32-low"
                  if config.thermochemical_state_high_low
                  else "float32-high-only"
              ),
              "space": "normalized-state",
              "low_equation": (
                  "float32((x64 - float64(float32(x64))) * low_scale)"
                  if config.thermochemical_state_high_low
                  else None
              ),
              "low_scale": config.thermochemical_state_low_scale,
              "consumers": (
                  ["auxiliary-temperature", "total-enthalpy"]
                  if config.thermochemical_state_high_low
                  else []
              ),
          },
          "thermochemical_process_feature_contract": {
              "mode": config.thermochemical_process_feature_mode,
              "physical_equation": (
                  "G = process_extent @ (2 * consumption + "
                  "process_stoich).T"
                  if config.thermochemical_process_feature_mode
                  == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
                  else (
                      "process_extent = demand * process_availability"
                      if config.thermochemical_process_feature_mode
                      == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
                      else None
                  )
              ),
              "feature_equation": (
                  "G ** 0.1"
                  if config.thermochemical_process_feature_mode
                  == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
                  else (
                      "where(process_extent == 0, 0, "
                      "clamp_min(process_extent, 1e-30) ** 0.1)"
                      if config.thermochemical_process_feature_mode
                      == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
                      else None
                  )
              ),
              "physical_dtype": "float64",
              "feature_dtype": "hidden dtype",
              "consumers": (
                  ["total-enthalpy"]
                  if config.thermochemical_process_feature_mode
                  != THERMOCHEMICAL_PROCESS_FEATURE_NONE
                  else []
              ),
              "auxiliary_temperature_uses_feature": False,
              "species_reaction_encoder_uses_feature": False,
          },
          "total_enthalpy_head_contract": {
              "hidden_dim_override": config.total_enthalpy_head_hidden_dim,
              "depth_override": config.total_enthalpy_head_depth,
              "resolved_hidden_dim": (
                  config.total_enthalpy_head_hidden_dim
                  or config.thermochemical_head_hidden_dim
                  or config.hidden_dim
              ),
              "resolved_depth": (
                  config.total_enthalpy_head_depth
                  or config.thermochemical_head_depth
              ),
              "auxiliary_temperature_head_uses_overrides": False,
          },
          "thermochemical_target_contract": (
              {
                  "mode": DELTA_H_TOTAL_MODE,
                  "label_backend": FLUENT_NATIVE_DI_BACKEND,
                  "before_dataset": "pairs/h_total_before",
                  "after_dataset": "pairs/h_total_after",
                  "target_equation": (
                      "delta_h_total = h_total_after - h_total_before"
                  ),
                  "target_units": "J/kg",
                  "total_enthalpy_delta_scale_j_per_kg": (
                      config.total_enthalpy_delta_scale
                  ),
              }
              if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
              else None
          ),
          "total_enthalpy_sign_balance_statistics": (
              {
                  "loss_weight": config.total_enthalpy_sign_balanced_loss_weight,
                  "training_sample_count": (
                      config.total_enthalpy_training_sample_count
                  ),
                  "negative_count": config.total_enthalpy_negative_count,
                  "positive_count": config.total_enthalpy_positive_count,
                  "negative_abs_target_sum": (
                      config.total_enthalpy_negative_abs_target_sum
                  ),
                  "positive_abs_target_sum": (
                      config.total_enthalpy_positive_abs_target_sum
                  ),
                  "objective": (
                      "0.5 * (negative absolute-error sum / negative "
                      "absolute-target sum + positive absolute-error sum / "
                      "positive absolute-target sum)"
                  ),
                  "zero_target_policy": "primary raw MAE only",
              }
              if config.total_enthalpy_sign_balanced_loss_weight > 0.0
              else None
          ),
          "training_config": asdict(config),
        "distributed_training": {
            "world_size": world_size,
            "batch_size_per_device": batch_size_per_device,
            "global_batch_size": world_size * batch_size_per_device,
            "learning_rate": learning_rate,
            "minimum_learning_rate": minimum_learning_rate,
            "scheduler": "cosine-annealing",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        "history": history,
    }


def main() -> None:
    args = parse_args()
    _validate_normalization_arguments(args)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = "nccl" if args.device == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.num_threads)
    device = (
        torch.device("cuda", local_rank)
        if args.device == "cuda"
        else torch.device("cpu")
    )
    deterministic = not args.non_deterministic
    _set_reproducibility(args.seed, deterministic)

    train_current, train_target, train_dt, train_species, train_attrs = (
        _load_core(args.train_source)
    )
    val_current, val_target, val_dt, val_species, val_attrs = _load_core(
        args.validation_source
    )
    if train_species != val_species:
        raise ValueError("Training and validation species orders differ")
    phase_name = _decode_phase(train_attrs, args.phase_name)
    gas = (
        ct.Solution(args.mechanism, phase_name)
        if phase_name
        else ct.Solution(args.mechanism)
    )
    if list(gas.species_names) != train_species:
        raise ValueError("Mechanism and dataset species orders differ")

    config = _training_config_from_args(args, world_size=world_size)
    if (
        config.thermochemical_delta_y_feature_mode
        == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
    ):
        config.thermochemical_delta_y_scales = tuple(
            compute_thermochemical_delta_y_scales(
                train_current,
                train_target,
                floor=config.thermochemical_delta_y_scale_floor,
            ).tolist()
        )
    _validate_thermochemical_configuration(config)
    _validate_row_tail_configuration(config)
    _require_deployment_label_sources(config, train_attrs, val_attrs)
    head_only_modes = sum(
        int(enabled)
        for enabled in (
            args.temperature_head_only,
            args.thermochemical_head_only,
            args.total_enthalpy_head_only,
            args.total_enthalpy_residual_head_only,
        )
    )
    if head_only_modes > 1:
        raise ValueError(
            "temperature-head-only, thermochemical-head-only, and "
            "total-enthalpy-head-only, and total-enthalpy-residual-head-only "
            "are mutually exclusive"
        )
    if args.total_enthalpy_residual_head_only and (
        config.total_enthalpy_residual_mode
        != TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
    ):
        raise ValueError(
            "total-enthalpy-residual-head-only requires signed-power "
            "total-enthalpy residual mode"
        )
    if config.total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE:
        if not args.total_enthalpy_residual_head_only:
            raise ValueError(
                "total-enthalpy residual mode requires "
                "--total-enthalpy-residual-head-only"
            )
        if not args.initial_checkpoint:
            raise ValueError(
                "total-enthalpy residual mode requires --initial-checkpoint"
            )
        if args.normalization_source != "initial-checkpoint":
            raise ValueError(
                "total-enthalpy residual mode requires normalization from "
                "the initial checkpoint"
            )
        if args.reset_thermochemical_head or args.reset_total_enthalpy_head:
            raise ValueError(
                "total-enthalpy residual mode cannot reset the physical "
                "baseline head"
            )
    if (
        args.total_enthalpy_head_only
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "total-enthalpy-head-only requires delta-h-total mode"
        )
    if args.reset_thermochemical_head and args.reset_total_enthalpy_head:
        raise ValueError(
            "reset-thermochemical-head and reset-total-enthalpy-head are "
            "mutually exclusive"
        )
    if args.reset_total_enthalpy_head and not args.initial_checkpoint:
        raise ValueError(
            "reset-total-enthalpy-head requires an initial checkpoint"
        )
    if (
        args.reset_total_enthalpy_head
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "reset-total-enthalpy-head requires delta-h-total mode"
        )
    if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE and (
        args.temperature_selection_weight != 0.0
        or args.temperature_head_only
    ):
        raise ValueError(
            "delta-h-total mode has no independent temperature head or "
            "temperature selection term"
        )
    train_delta_h_total = val_delta_h_total = None
    if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE:
        _, _, train_delta_h_total = _load_total_enthalpy_increment_labels(
            args.train_source, expected_rows=train_current.shape[0]
        )
        _, _, val_delta_h_total = _load_total_enthalpy_increment_labels(
            args.validation_source, expected_rows=val_current.shape[0]
        )
        if config.total_enthalpy_sign_balanced_loss_weight > 0.0:
            sign_statistics = compute_total_enthalpy_sign_statistics(
                train_delta_h_total
            )
            config.total_enthalpy_negative_abs_target_sum = float(
                sign_statistics["negative_abs_target_sum"]
            )
            config.total_enthalpy_positive_abs_target_sum = float(
                sign_statistics["positive_abs_target_sum"]
            )
            config.total_enthalpy_negative_count = int(
                sign_statistics["negative_count"]
            )
            config.total_enthalpy_positive_count = int(
                sign_statistics["positive_count"]
            )
            config.total_enthalpy_training_sample_count = int(
                sign_statistics["training_sample_count"]
            )
    jvp_training_data = _prepare_jvp_training_data(
        config,
        species_names=gas.species_names,
        train_dt=train_dt,
        validation_dt=val_dt,
    )
    jvp_loss_config = _jvp_loss_config(config)
    deployment_loss_config = _deployment_loss_config(config)

    initial_checkpoint = None
    if args.initial_checkpoint:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
    if config.total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE:
        initial_config = initial_checkpoint.get("training_config", {})
        initial_target_transform = str(
            initial_config.get(
                "total_enthalpy_target_transform",
                TOTAL_ENTHALPY_TARGET_IDENTITY,
            )
        )
        if initial_target_transform != TOTAL_ENTHALPY_TARGET_IDENTITY:
            raise ValueError(
                "total-enthalpy residual mode requires an identity-transform "
                "baseline checkpoint"
            )
    if args.normalization_source == "initial-checkpoint":
        state_mean, state_std, log_dt_mean, log_dt_std = (
            _normalizers_from_initial_checkpoint(
                initial_checkpoint,
                state_dimension=train_current.shape[1],
                checkpoint_path=args.initial_checkpoint,
            )
        )
        initial_training_source = initial_checkpoint.get(
            "source_path", initial_checkpoint.get("train_source")
        )
        initial_epoch = initial_checkpoint.get("epoch")
        normalization_provenance = {
            "source": "initial-checkpoint",
            "initial_checkpoint": str(args.initial_checkpoint),
            "initial_checkpoint_epoch": (
                int(initial_epoch) if initial_epoch is not None else None
            ),
            "initial_checkpoint_normalization_source": str(
                initial_checkpoint.get(
                    "normalization_source", "legacy-unspecified"
                )
            ),
            "initial_checkpoint_training_source": (
                str(initial_training_source)
                if initial_training_source is not None
                else None
            ),
        }
    else:
        state_mean = train_current.mean(axis=0, dtype=np.float64)
        state_std = train_current.std(axis=0, dtype=np.float64)
        state_std = np.where(state_std > 0.0, state_std, 1.0)
        train_log_dt = np.log(np.maximum(train_dt, 1.0e-300))
        log_dt_mean = np.asarray([train_log_dt.mean()], dtype=np.float64)
        log_dt_std = np.asarray([train_log_dt.std()], dtype=np.float64)
        log_dt_std = np.where(log_dt_std > 1.0e-8, log_dt_std, 1.0)
        normalization_provenance = {
            "source": "train",
            "train_source": str(args.train_source),
        }
    train_x, train_x_low, train_t = _normalize_inputs(
        train_current,
        train_dt,
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
        high_low=config.thermochemical_state_high_low,
        low_scale=config.thermochemical_state_low_scale,
    )
    val_x, val_x_low, val_t = _normalize_inputs(
        val_current,
        val_dt,
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
        high_low=config.thermochemical_state_high_low,
        low_scale=config.thermochemical_state_low_scale,
    )

    model = _build_model(config, gas, train_current.shape[1]).to(device)
    if initial_checkpoint is not None:
        initial = initial_checkpoint
        initial_state = dict(initial["net"])
        exact_missing_keys = None
        if config.total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE:
            initial_config = initial.get("training_config", {})
            initial_residual_mode = str(
                initial_config.get(
                    "total_enthalpy_residual_mode",
                    TOTAL_ENTHALPY_RESIDUAL_NONE,
                )
            )
            if initial_residual_mode == TOTAL_ENTHALPY_RESIDUAL_NONE:
                exact_missing_keys = {
                    name
                    for name in model.state_dict()
                    if name.startswith("total_enthalpy_residual_head.")
                }
            elif initial_residual_mode == config.total_enthalpy_residual_mode:
                initial_residual_alpha = float(
                    initial_config.get("total_enthalpy_residual_alpha", 0.1)
                )
                if initial_residual_alpha != config.total_enthalpy_residual_alpha:
                    raise ValueError(
                        "cannot change total-enthalpy residual alpha while "
                        "continuing a residual checkpoint"
                    )
                exact_missing_keys = set()
            else:
                raise ValueError(
                    "unsupported total-enthalpy residual migration: "
                    f"{initial_residual_mode!r} -> "
                    f"{config.total_enthalpy_residual_mode!r}"
                )
        elif args.reset_thermochemical_head:
            head_prefixes = [(
                "total_enthalpy_delta_head."
                if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
                else "temperature_delta_head."
            )]
            if config.thermochemical_aux_temperature:
                head_prefixes.append(
                    "thermochemical_aux_temperature_head."
                )
            initial_state = {
                name: value
                for name, value in initial_state.items()
                if not name.startswith(tuple(head_prefixes))
            }
        elif args.reset_total_enthalpy_head:
            total_enthalpy_head_prefix = "total_enthalpy_delta_head."
            exact_missing_keys = {
                name
                for name in model.state_dict()
                if name.startswith(total_enthalpy_head_prefix)
            }
            initial_state = {
                name: value
                for name, value in initial_state.items()
                if not name.startswith(total_enthalpy_head_prefix)
            }
        else:
            initial_config = initial.get("training_config", {})
            initial_process_feature_mode = str(
                initial.get(
                    "thermochemical_process_feature_mode",
                    initial_config.get(
                        "thermochemical_process_feature_mode",
                        THERMOCHEMICAL_PROCESS_FEATURE_NONE,
                    ),
                )
            )
            target_process_feature_mode = (
                config.thermochemical_process_feature_mode
            )
            if (
                initial_process_feature_mode != target_process_feature_mode
                and not (
                    initial_process_feature_mode
                    == THERMOCHEMICAL_PROCESS_FEATURE_NONE
                    and target_process_feature_mode
                    in {
                        THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER,
                        THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS,
                    }
                )
            ):
                raise ValueError(
                    "unsupported thermochemical process-feature migration: "
                    f"{initial_process_feature_mode!r} -> "
                    f"{target_process_feature_mode!r}"
                )
            target_state = model.state_dict()
            migratable_weights = []
            if config.thermochemical_state_high_low:
                migratable_weights.append(
                    "thermochemical_aux_temperature_head.0.weight"
                )
                migratable_weights.append(
                    "total_enthalpy_delta_head.0.weight"
                )
            if initial_process_feature_mode != target_process_feature_mode:
                migratable_weights.append(
                    "total_enthalpy_delta_head.0.weight"
                )
            for name in dict.fromkeys(migratable_weights):
                if name not in initial_state or name not in target_state:
                    continue
                source_weight = initial_state[name]
                target_weight = target_state[name]
                if source_weight.shape == target_weight.shape:
                    continue
                if (
                    source_weight.ndim != 2
                    or target_weight.ndim != 2
                    or source_weight.shape[0] != target_weight.shape[0]
                    or source_weight.shape[1] > target_weight.shape[1]
                ):
                    raise ValueError(
                        f"cannot append thermochemical features to {name}: "
                        f"{tuple(source_weight.shape)} -> "
                        f"{tuple(target_weight.shape)}"
                    )
                expanded = torch.zeros_like(target_weight)
                expanded[:, : source_weight.shape[1]] = source_weight
                initial_state[name] = expanded
        incompatible = model.load_state_dict(initial_state, strict=False)
        if exact_missing_keys is not None:
            actual_missing_keys = set(incompatible.missing_keys)
            actual_unexpected_keys = set(incompatible.unexpected_keys)
            if (
                actual_missing_keys != exact_missing_keys
                or actual_unexpected_keys
            ):
                raise ValueError(
                    "checkpoint migration requires an exact load except for "
                    "the explicitly permitted head keys; expected missing keys "
                    f"{sorted(exact_missing_keys)}, got missing "
                    f"{sorted(actual_missing_keys)} and unexpected "
                    f"{sorted(actual_unexpected_keys)}"
                )
    if args.temperature_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.temperature_delta_head.parameters():
            parameter.requires_grad_(True)
    elif args.thermochemical_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        thermochemical_head = (
            model.total_enthalpy_delta_head
            if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
            else model.temperature_delta_head
        )
        for parameter in thermochemical_head.parameters():
            parameter.requires_grad_(True)
        if config.thermochemical_aux_temperature:
            for parameter in (
                model.thermochemical_aux_temperature_head.parameters()
            ):
                parameter.requires_grad_(True)
    elif args.total_enthalpy_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.total_enthalpy_delta_head.parameters():
            parameter.requires_grad_(True)
    elif args.total_enthalpy_residual_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.total_enthalpy_residual_head.parameters():
            parameter.requires_grad_(True)
    head_only_training = bool(
        args.temperature_head_only
        or args.thermochemical_head_only
        or args.total_enthalpy_head_only
        or args.total_enthalpy_residual_head_only
    )
    jvp_optimization_enabled = bool(
        jvp_training_data is not None
        and config.source_jvp_loss_weight > 0.0
        and not head_only_training
    )
    if args.device == "cuda":
        distributed_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=jvp_training_data is None,
            find_unused_parameters=(
                jvp_training_data is not None and not head_only_training
            ),
        )
    else:
        distributed_model = DistributedDataParallel(
            model,
            broadcast_buffers=jvp_training_data is None,
            find_unused_parameters=(
                jvp_training_data is not None and not head_only_training
            ),
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in distributed_model.parameters()
         if parameter.requires_grad),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )

    output = Path(args.output)
    final_output = Path(args.final_output or output.with_name("final.pt"))
    resume_output = Path(args.resume_output or output.with_name("last.pt"))
    history: list[dict] = []
    start_epoch = 1
    best_epoch = 0
    best_validation_loss = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["net"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume.get("best_epoch", 0))
        best_validation_loss = float(
            resume.get("best_validation_loss", float("inf"))
        )

    state_mean_t = torch.from_numpy(state_mean).to(
        device=device, dtype=torch.float64
    )
    state_std_t = torch.from_numpy(state_std).to(
        device=device, dtype=torch.float64
    )
    formation_enthalpy_t = torch.as_tensor(
        [
            gas.species(index).thermo.h(298.15) / gas.molecular_weights[index]
            for index in range(gas.n_species)
        ],
        dtype=torch.float64,
        device=device,
    )
    molecular_weights_t = torch.as_tensor(
        gas.molecular_weights, dtype=torch.float64, device=device
    )
    zero_species_enthalpy_t = torch.zeros(
        gas.n_species, dtype=torch.float64, device=device
    )
    usable = (train_current.shape[0] // world_size) * world_size
    local_count = usable // world_size
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "training_start",
                    "train_samples": int(train_current.shape[0]),
                    "validation_samples": int(val_current.shape[0]),
                    "usable_train_samples": int(usable),
                    "world_size": world_size,
                    "batch_size_per_device": args.batch_size_per_device,
                    "global_batch_size": (
                        args.batch_size_per_device * world_size
                    ),
                    "epochs": args.epochs,
                    "start_epoch": start_epoch,
                    "scheduler": "cosine-annealing",
                    "minimum_learning_rate": args.minimum_learning_rate,
                    "normalization_source": args.normalization_source,
                    "normalization_provenance": normalization_provenance,
                    "deterministic": deterministic,
                    "device": str(device),
                    "thermochemical_output_mode": (
                        config.thermochemical_output_mode
                    ),
                    "thermochemical_head_input_mode": (
                        config.thermochemical_head_input_mode
                    ),
                    "thermochemical_head_hidden_dim": (
                        config.thermochemical_head_hidden_dim
                    ),
                    "thermochemical_head_depth": config.thermochemical_head_depth,
                    "total_enthalpy_head_hidden_dim": (
                        config.total_enthalpy_head_hidden_dim
                    ),
                    "total_enthalpy_head_depth": (
                        config.total_enthalpy_head_depth
                    ),
                    "thermochemical_aux_temperature": (
                        config.thermochemical_aux_temperature
                    ),
                    "thermochemical_aux_temperature_loss_weight": (
                        config.thermochemical_aux_temperature_loss_weight
                    ),
                    "thermochemical_process_feature_mode": (
                        config.thermochemical_process_feature_mode
                    ),
                    "thermochemical_state_high_low": (
                        config.thermochemical_state_high_low
                    ),
                    "thermochemical_state_low_scale": (
                        config.thermochemical_state_low_scale
                    ),
                    "reset_thermochemical_head": args.reset_thermochemical_head,
                    "reset_total_enthalpy_head": (
                        args.reset_total_enthalpy_head
                    ),
                      "total_enthalpy_head_only": args.total_enthalpy_head_only,
                      "total_enthalpy_residual_head_only": (
                          args.total_enthalpy_residual_head_only
                      ),
                      "total_enthalpy_residual_mode": (
                          config.total_enthalpy_residual_mode
                      ),
                      "total_enthalpy_residual_alpha": (
                          config.total_enthalpy_residual_alpha
                      ),
                      "total_enthalpy_residual_transformed_loss_weight": (
                          config.total_enthalpy_residual_transformed_loss_weight
                      ),
                    "total_enthalpy_delta_scale": (
                        config.total_enthalpy_delta_scale
                    ),
                    "total_enthalpy_delta_loss_weight": (
                        config.total_enthalpy_delta_loss_weight
                    ),
                    "jvp_dataset": config.jvp_dataset,
                    "jvp_pairs": (
                        int(jvp_training_data.batch.x.shape[0])
                        if jvp_training_data is not None
                        else 0
                    ),
                    "jvp_optimization_enabled": jvp_optimization_enabled,
                    "source_jvp_loss_weight": (
                        config.source_jvp_loss_weight
                    ),
                    "source_jvp_loss_scale": config.source_jvp_loss_scale,
                    "source_loss_weight": config.source_loss_weight,
                    "source_loss_scale": config.source_loss_scale,
                    "row_tail_relative_delta_loss_weight": (
                        config.row_tail_relative_delta_loss_weight
                    ),
                    "row_tail_relative_delta_fraction": (
                        config.row_tail_relative_delta_fraction
                    ),
                    "row_tail_relative_delta_floor": (
                        config.row_tail_relative_delta_floor
                    ),
                    "row_tail_relative_delta_selection_weight": (
                        config.row_tail_relative_delta_selection_weight
                    ),
                    "mixture_molecular_weight_loss_weight": (
                        config.mixture_molecular_weight_loss_weight
                    ),
                    "mixture_molecular_weight_loss_scale": (
                        config.mixture_molecular_weight_loss_scale
                    ),
                    "density_increment_loss_weight": (
                        config.density_increment_loss_weight
                    ),
                    "density_increment_loss_scale": (
                        config.density_increment_loss_scale
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        epoch_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, epoch])
        )
        order = epoch_rng.permutation(train_current.shape[0])[:usable]
        local_order = order[rank::world_size]
        if local_order.size != local_count:
            raise RuntimeError("Unequal distributed sample partition")

        distributed_model.train()
        local_loss_sum = torch.zeros(14, dtype=torch.float64, device=device)
        local_samples = 0
        for start in range(0, local_count, args.batch_size_per_device):
            indices = local_order[start : start + args.batch_size_per_device]
            x = torch.from_numpy(train_x[indices]).to(device)
            x_low = (
                torch.from_numpy(train_x_low[indices]).to(device)
                if train_x_low is not None
                else None
            )
            t = torch.from_numpy(train_t[indices]).to(device)
            y0 = torch.from_numpy(train_current[indices, 2:]).to(device)
            target = torch.from_numpy(train_target[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            target_delta_h_total = (
                torch.from_numpy(train_delta_h_total[indices]).to(device)
                if train_delta_h_total is not None
                else None
            )
            dt_batch = torch.from_numpy(train_dt[indices]).to(device)
            loss, components, _, _ = _batch_loss(
                distributed_model,
                x,
                x_low,
                t,
                y0,
                target,
                state_mean_t,
                state_std_t,
                args.transform_alpha,
                args.tp_loss_weight,
                args.state_loss_weight,
                args.delta_loss_weight,
                args.relative_delta_loss_weight,
                args.row_tail_relative_delta_loss_weight,
                args.row_tail_relative_delta_fraction,
                args.row_tail_relative_delta_floor,
                formation_enthalpy_t,
                args.formation_energy_loss_weight,
                args.direction_loss_weight,
                args.opposing_extent_loss_weight,
                args.temperature_delta_scale,
                args.temperature_rmse_weight,
                config.thermochemical_output_mode,
                config.total_enthalpy_delta_scale,
                config.total_enthalpy_delta_loss_weight,
                config.total_enthalpy_target_transform,
                  config.total_enthalpy_transform_alpha,
                  config.total_enthalpy_transformed_loss_weight,
                  config.total_enthalpy_residual_mode,
                  config.total_enthalpy_residual_alpha,
                  config.total_enthalpy_residual_transformed_loss_weight,
                config.total_enthalpy_sign_balanced_loss_weight,
                config.total_enthalpy_negative_abs_target_sum,
                config.total_enthalpy_positive_abs_target_sum,
                config.total_enthalpy_training_sample_count,
                config.thermochemical_aux_temperature_loss_weight,
                config.thermochemical_state_low_scale,
                target_delta_h_total,
                args.extent_scale,
                dt_batch,
                molecular_weights_t,
                zero_species_enthalpy_t,
                deployment_loss_config,
            )
            loss.backward()
            optimizer.step()
            count = indices.size
            local_loss_sum[0] += loss.detach().double() * count
            for component_index, component in enumerate(components, start=1):
                local_loss_sum[component_index] += (
                    component.detach().double() * count
                )
            local_samples += count

        if jvp_optimization_enabled:
            jvp_batches = deterministic_paired_batch_indices(
                jvp_training_data.train_indices,
                batch_size=args.batch_size_per_device,
                seed=args.seed,
                epoch=epoch,
                rank=rank,
                world_size=world_size,
            )
            for pair_indices in jvp_batches:
                optimizer.zero_grad(set_to_none=True)
                jvp_report = _distributed_jvp_loss_for_indices(
                    distributed_model,
                    jvp_training_data,
                    pair_indices,
                    config=config,
                    loss_config=jvp_loss_config,
                    state_mean=state_mean_t,
                    state_std=state_std_t,
                    log_dt_mean=torch.as_tensor(
                        log_dt_mean, dtype=torch.float64, device=device
                    ),
                    log_dt_std=torch.as_tensor(
                        log_dt_std, dtype=torch.float64, device=device
                    ),
                    device=device,
                )
                jvp_report["total"].backward()
                optimizer.step()

        scheduler.step()
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        sample_count = torch.tensor(
            [local_samples], dtype=torch.float64, device=device
        )
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        train_losses = (local_loss_sum / sample_count[0]).cpu().numpy()
        training_seconds = time.perf_counter() - epoch_started

        should_validate = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        )
        validation_metrics = torch.zeros(21, dtype=torch.float64, device=device)
        jvp_metrics: dict[str, float] = {}
        validation_started = time.perf_counter()
        if should_validate:
            dist.barrier()
            if rank == 0:
                model.eval()
                total = 0.0
                count = 0
                absolute_error = 0.0
                target_magnitude = 0.0
                delta_dot = 0.0
                predicted_square = 0.0
                target_square = 0.0
                temperature_absolute_error = 0.0
                temperature_target_magnitude = 0.0
                temperature_squared_error = 0.0
                temperature_target_squared = 0.0
                total_enthalpy_absolute_error = 0.0
                total_enthalpy_target_magnitude = 0.0
                total_enthalpy_negative_absolute_error = 0.0
                total_enthalpy_negative_target_magnitude = 0.0
                total_enthalpy_positive_absolute_error = 0.0
                total_enthalpy_positive_target_magnitude = 0.0
                auxiliary_temperature_scaled_total = 0.0
                deployment_scaled_sum = np.zeros(3, dtype=np.float64)
                row_relative_delta_values = []
                with torch.inference_mode():
                    for start in range(
                        0, val_current.shape[0], args.batch_size_per_device
                    ):
                        stop = min(
                            start + args.batch_size_per_device,
                            val_current.shape[0],
                        )
                        x = torch.from_numpy(val_x[start:stop]).to(device)
                        x_low = (
                            torch.from_numpy(val_x_low[start:stop]).to(device)
                            if val_x_low is not None
                            else None
                        )
                        t = torch.from_numpy(val_t[start:stop]).to(device)
                        y0 = torch.from_numpy(
                            val_current[start:stop, 2:]
                        ).to(device)
                        target = torch.from_numpy(
                            val_target[start:stop]
                        ).to(device)
                        target_delta_h_total = (
                            torch.from_numpy(
                                val_delta_h_total[start:stop]
                            ).to(device)
                            if val_delta_h_total is not None
                            else None
                        )
                        dt_batch = torch.from_numpy(val_dt[start:stop]).to(device)
                        loss, components, prediction, thermochemical_prediction = _batch_loss(
                            model,
                            x,
                            x_low,
                            t,
                            y0,
                            target,
                            state_mean_t,
                            state_std_t,
                            args.transform_alpha,
                            args.tp_loss_weight,
                            args.state_loss_weight,
                            args.delta_loss_weight,
                            args.relative_delta_loss_weight,
                            args.row_tail_relative_delta_loss_weight,
                            args.row_tail_relative_delta_fraction,
                            args.row_tail_relative_delta_floor,
                            formation_enthalpy_t,
                            args.formation_energy_loss_weight,
                            args.direction_loss_weight,
                            args.opposing_extent_loss_weight,
                            args.temperature_delta_scale,
                            args.temperature_rmse_weight,
                            config.thermochemical_output_mode,
                            config.total_enthalpy_delta_scale,
                            config.total_enthalpy_delta_loss_weight,
                            config.total_enthalpy_target_transform,
                              config.total_enthalpy_transform_alpha,
                              config.total_enthalpy_transformed_loss_weight,
                              config.total_enthalpy_residual_mode,
                              config.total_enthalpy_residual_alpha,
                              config.total_enthalpy_residual_transformed_loss_weight,
                            config.total_enthalpy_sign_balanced_loss_weight,
                            config.total_enthalpy_negative_abs_target_sum,
                            config.total_enthalpy_positive_abs_target_sum,
                            config.total_enthalpy_training_sample_count,
                            config.thermochemical_aux_temperature_loss_weight,
                            config.thermochemical_state_low_scale,
                            target_delta_h_total,
                            args.extent_scale,
                            dt_batch,
                            molecular_weights_t,
                            zero_species_enthalpy_t,
                            deployment_loss_config,
                        )
                        predicted_delta = prediction[:, 2:] - y0
                        target_delta = target[:, 2:] - y0
                        row_relative_delta_values.append(
                            _row_relative_delta_errors(
                                predicted_delta,
                                target_delta,
                                config.row_tail_relative_delta_floor,
                            ).cpu().numpy()
                        )
                        absolute_error += float(torch.sum(torch.abs(predicted_delta - target_delta)))
                        target_magnitude += float(torch.sum(torch.abs(target_delta)))
                        delta_dot += float(torch.sum(predicted_delta * target_delta))
                        predicted_square += float(torch.sum(predicted_delta.square()))
                        target_square += float(torch.sum(target_delta.square()))
                        if config.thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
                            predicted_temperature = (
                                prediction[:, 0] * state_std_t[0]
                                + state_mean_t[0]
                            )
                            temperature_absolute_error += float(
                                torch.sum(
                                    torch.abs(
                                        predicted_temperature - target[:, 0]
                                    )
                                )
                            )
                            temperature_squared_error += float(
                                torch.sum(
                                    (
                                        predicted_temperature
                                        - target[:, 0]
                                    ).square()
                                )
                            )
                            current_validation_temperature = torch.from_numpy(
                                val_current[start:stop, 0]
                            ).to(device)
                            temperature_target_magnitude += float(
                                torch.sum(
                                    torch.abs(
                                        target[:, 0]
                                        - current_validation_temperature
                                    )
                                )
                            )
                            temperature_target_squared += float(
                                torch.sum(
                                    (
                                        target[:, 0]
                                        - current_validation_temperature
                                    ).square()
                                )
                            )
                        else:
                            enthalpy_absolute_error = torch.abs(
                                thermochemical_prediction
                                - target_delta_h_total
                            )
                            total_enthalpy_absolute_error += float(
                                torch.sum(enthalpy_absolute_error)
                            )
                            total_enthalpy_target_magnitude += float(
                                torch.sum(torch.abs(target_delta_h_total))
                            )
                            negative_target = target_delta_h_total < 0.0
                            positive_target = target_delta_h_total > 0.0
                            total_enthalpy_negative_absolute_error += float(
                                torch.sum(enthalpy_absolute_error[negative_target])
                            )
                            total_enthalpy_negative_target_magnitude += float(
                                torch.sum(
                                    torch.abs(
                                        target_delta_h_total[negative_target]
                                    )
                                )
                            )
                            total_enthalpy_positive_absolute_error += float(
                                torch.sum(enthalpy_absolute_error[positive_target])
                            )
                            total_enthalpy_positive_target_magnitude += float(
                                torch.sum(
                                    torch.abs(
                                        target_delta_h_total[positive_target]
                                    )
                                )
                            )
                        batch_count = stop - start
                        auxiliary_temperature_scaled_total += float(
                            components[9]
                        ) * batch_count
                        for deployment_index in range(3):
                            deployment_scaled_sum[deployment_index] += float(
                                components[10 + deployment_index]
                            ) * batch_count
                        total += float(loss) * batch_count
                        count += batch_count
                normalized_delta_mae = absolute_error / max(target_magnitude, 1.0e-300)
                global_delta_cosine = delta_dot / max(
                    (predicted_square * target_square) ** 0.5, 1.0e-300
                )
                selection_score = normalized_delta_mae + 0.25 * (1.0 - global_delta_cosine)
                temperature_delta_nmae = temperature_absolute_error / max(
                    temperature_target_magnitude, 1.0e-300
                )
                temperature_delta_nrmse = (
                    temperature_squared_error
                    / max(temperature_target_squared, 1.0e-300)
                ) ** 0.5
                total_enthalpy_delta_mae = (
                    total_enthalpy_absolute_error / max(count, 1)
                )
                total_enthalpy_delta_fixed_scale_loss = (
                    total_enthalpy_delta_mae
                    / config.total_enthalpy_delta_scale
                )
                total_enthalpy_delta_nmae = (
                    total_enthalpy_absolute_error
                    / max(total_enthalpy_target_magnitude, 1.0e-300)
                )
                total_enthalpy_negative_nmae = (
                    total_enthalpy_negative_absolute_error
                    / max(
                        total_enthalpy_negative_target_magnitude,
                        1.0e-300,
                    )
                )
                total_enthalpy_positive_nmae = (
                    total_enthalpy_positive_absolute_error
                    / max(
                        total_enthalpy_positive_target_magnitude,
                        1.0e-300,
                    )
                )
                total_enthalpy_negative_error_fraction = (
                    total_enthalpy_negative_absolute_error
                    / max(total_enthalpy_absolute_error, 1.0e-300)
                )
                total_enthalpy_positive_error_fraction = (
                    total_enthalpy_positive_absolute_error
                    / max(total_enthalpy_absolute_error, 1.0e-300)
                )
                row_relative_delta = np.concatenate(row_relative_delta_values)
                row_tail_count = min(
                    row_relative_delta.size,
                    max(
                        1,
                        int(
                            np.ceil(
                                config.row_tail_relative_delta_fraction
                                * row_relative_delta.size
                            )
                        ),
                    ),
                )
                row_tail = np.partition(
                    row_relative_delta, row_relative_delta.size - row_tail_count
                )[-row_tail_count:]
                row_tail_relative_delta_loss = float(
                    np.mean(np.log1p(row_tail))
                )
                row_relative_delta_p50, row_relative_delta_p90, row_relative_delta_p99 = (
                    np.quantile(row_relative_delta, (0.5, 0.9, 0.99))
                )
                row_relative_delta_max = float(np.max(row_relative_delta))
                validation_metrics[:] = torch.tensor(
                    [total / count, normalized_delta_mae, global_delta_cosine,
                     temperature_delta_nmae, temperature_delta_nrmse,
                     total_enthalpy_delta_mae,
                     total_enthalpy_delta_fixed_scale_loss,
                     total_enthalpy_delta_nmae,
                     auxiliary_temperature_scaled_total / max(count, 1),
                     *(deployment_scaled_sum / max(count, 1)),
                     row_tail_relative_delta_loss,
                     row_relative_delta_p50,
                     row_relative_delta_p90,
                     row_relative_delta_p99,
                     row_relative_delta_max,
                     total_enthalpy_negative_nmae,
                     total_enthalpy_positive_nmae,
                     total_enthalpy_negative_error_fraction,
                     total_enthalpy_positive_error_fraction],
                    dtype=torch.float64,
                    device=device,
                )
            dist.broadcast(validation_metrics, src=0)
            if jvp_training_data is not None:
                jvp_metrics.update(
                    _evaluate_jvp_partition(
                        model,
                        jvp_training_data,
                        jvp_training_data.train_indices,
                        prefix="train",
                        config=config,
                        loss_config=jvp_loss_config,
                        state_mean=state_mean_t,
                        state_std=state_std_t,
                        log_dt_mean=torch.as_tensor(
                            log_dt_mean, dtype=torch.float64, device=device
                        ),
                        log_dt_std=torch.as_tensor(
                            log_dt_std, dtype=torch.float64, device=device
                        ),
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        base_affinity=None,
                        perturbed_affinity=None,
                    )
                )
                jvp_metrics.update(
                    _evaluate_jvp_partition(
                        model,
                        jvp_training_data,
                        jvp_training_data.validation_indices,
                        prefix="validation",
                        config=config,
                        loss_config=jvp_loss_config,
                        state_mean=state_mean_t,
                        state_std=state_std_t,
                        log_dt_mean=torch.as_tensor(
                            log_dt_mean, dtype=torch.float64, device=device
                        ),
                        log_dt_std=torch.as_tensor(
                            log_dt_std, dtype=torch.float64, device=device
                        ),
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        base_affinity=None,
                        perturbed_affinity=None,
                    )
                )

        validation_seconds = (
            time.perf_counter() - validation_started
            if should_validate
            else 0.0
        )
        if rank == 0 and should_validate:
            transformed_value = float(validation_metrics[0])
            normalized_delta_mae = float(validation_metrics[1])
            global_delta_cosine = float(validation_metrics[2])
            temperature_delta_nmae = float(validation_metrics[3])
            temperature_delta_nrmse = float(validation_metrics[4])
            total_enthalpy_delta_mae = float(validation_metrics[5])
            total_enthalpy_delta_fixed_scale_loss = float(
                validation_metrics[6]
            )
            total_enthalpy_delta_nmae = float(validation_metrics[7])
            auxiliary_temperature_scaled_loss = float(
                validation_metrics[8]
            )
            source_fixed_scale_loss = float(validation_metrics[9])
            mixture_molecular_weight_fixed_scale_loss = float(
                validation_metrics[10]
            )
            density_increment_fixed_scale_loss = float(validation_metrics[11])
            row_tail_relative_delta_loss = float(validation_metrics[12])
            row_relative_delta_p50 = float(validation_metrics[13])
            row_relative_delta_p90 = float(validation_metrics[14])
            row_relative_delta_p99 = float(validation_metrics[15])
            row_relative_delta_max = float(validation_metrics[16])
            total_enthalpy_negative_nmae = float(validation_metrics[17])
            total_enthalpy_positive_nmae = float(validation_metrics[18])
            total_enthalpy_negative_error_fraction = float(
                validation_metrics[19]
            )
            total_enthalpy_positive_error_fraction = float(
                validation_metrics[20]
            )
            total_enthalpy_selection = (
                config.total_enthalpy_delta_loss_weight
                * total_enthalpy_delta_fixed_scale_loss
                if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
                else 0.0
            )
            source_jvp_selection = (
                config.source_jvp_loss_weight
                * jvp_metrics.get(
                    "validation_source_jvp_fixed_scale_loss", 0.0
                )
            )
            row_tail_relative_delta_selection = (
                config.row_tail_relative_delta_selection_weight
                * row_tail_relative_delta_loss
            )
            value = (
                normalized_delta_mae
                + 0.25 * (1.0 - global_delta_cosine)
                + args.temperature_selection_weight * temperature_delta_nmae
                + args.temperature_rmse_selection_weight
                * temperature_delta_nrmse
                + total_enthalpy_selection
                + source_jvp_selection
                + config.source_loss_weight * source_fixed_scale_loss
                + config.mixture_molecular_weight_loss_weight
                * mixture_molecular_weight_fixed_scale_loss
                + config.density_increment_loss_weight
                * density_increment_fixed_scale_loss
                + row_tail_relative_delta_selection
            )
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": float(train_losses[0]),
                "train_tp_loss": float(train_losses[1]),
                "train_y_loss": float(train_losses[2]),
                "train_delta_loss": float(train_losses[3]),
                "train_relative_delta_loss": float(train_losses[4]),
                "train_row_tail_relative_delta_loss": float(train_losses[5]),
                "train_formation_energy_loss": float(train_losses[6]),
                "train_direction_loss": float(train_losses[7]),
                "train_opposing_extent_loss": float(train_losses[8]),
                "train_total_enthalpy_delta_fixed_scale_loss": float(
                    train_losses[9]
                ),
                "train_auxiliary_temperature_scaled_loss": float(
                    train_losses[10]
                ),
                "train_source_fixed_scale_loss": float(train_losses[11]),
                "train_mixture_molecular_weight_fixed_scale_loss": float(
                    train_losses[12]
                ),
                "train_density_increment_fixed_scale_loss": float(
                    train_losses[13]
                ),
                "validation_loss": transformed_value,
                "validation_normalized_delta_mae": normalized_delta_mae,
                "validation_global_delta_cosine": global_delta_cosine,
                "validation_temperature_delta_nmae": temperature_delta_nmae,
                "validation_temperature_delta_nrmse": (
                    temperature_delta_nrmse
                ),
                "validation_total_enthalpy_delta_mae_j_per_kg": (
                    total_enthalpy_delta_mae
                ),
                "validation_total_enthalpy_delta_fixed_scale_loss": (
                    total_enthalpy_delta_fixed_scale_loss
                ),
                "validation_total_enthalpy_delta_nmae": (
                    total_enthalpy_delta_nmae
                ),
                "validation_total_enthalpy_negative_nmae": (
                    total_enthalpy_negative_nmae
                ),
                "validation_total_enthalpy_positive_nmae": (
                    total_enthalpy_positive_nmae
                ),
                "validation_total_enthalpy_negative_error_fraction": (
                    total_enthalpy_negative_error_fraction
                ),
                "validation_total_enthalpy_positive_error_fraction": (
                    total_enthalpy_positive_error_fraction
                ),
                "validation_auxiliary_temperature_scaled_loss": (
                    auxiliary_temperature_scaled_loss
                ),
                "validation_total_enthalpy_selection_contribution": (
                    total_enthalpy_selection
                ),
                "validation_source_jvp_selection_contribution": (
                    source_jvp_selection
                ),
                "validation_row_tail_relative_delta_loss": (
                    row_tail_relative_delta_loss
                ),
                "validation_row_tail_relative_delta_selection_contribution": (
                    row_tail_relative_delta_selection
                ),
                "validation_row_relative_delta_p50": row_relative_delta_p50,
                "validation_row_relative_delta_p90": row_relative_delta_p90,
                "validation_row_relative_delta_p99": row_relative_delta_p99,
                "validation_row_relative_delta_max": row_relative_delta_max,
                "validation_source_mae_per_s": (
                    source_fixed_scale_loss * config.source_loss_scale
                ),
                "validation_source_fixed_scale_loss": source_fixed_scale_loss,
                "validation_mixture_molecular_weight_mae_kg_per_kmol": (
                    mixture_molecular_weight_fixed_scale_loss
                    * config.mixture_molecular_weight_loss_scale
                ),
                "validation_mixture_molecular_weight_fixed_scale_loss": (
                    mixture_molecular_weight_fixed_scale_loss
                ),
                "validation_density_increment_mae_kg_per_m3": (
                    density_increment_fixed_scale_loss
                    * config.density_increment_loss_scale
                ),
                "validation_density_increment_fixed_scale_loss": (
                    density_increment_fixed_scale_loss
                ),
                "validation_selection_score": value,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "training_samples_per_second": (
                    usable / training_seconds if training_seconds > 0.0 else None
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            record.update(jvp_metrics)
            history.append(record)
            if value < best_validation_loss:
                best_validation_loss = value
                best_epoch = epoch
                checkpoint = _make_checkpoint(
                    model=model,
                    config=config,
                    state_mean=state_mean,
                    state_std=state_std,
                    log_dt_mean=log_dt_mean,
                    log_dt_std=log_dt_std,
                    gas=gas,
                    train_source=args.train_source,
                    validation_source=args.validation_source,
                    mechanism=args.mechanism,
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    world_size=world_size,
                    batch_size_per_device=args.batch_size_per_device,
                    learning_rate=args.learning_rate,
                    minimum_learning_rate=args.minimum_learning_rate,
                    normalization_source=args.normalization_source,
                    normalization_provenance=normalization_provenance,
                )
                _atomic_save(checkpoint, output)
                record["saved_best"] = True
            print(json.dumps(record, sort_keys=True), flush=True)

        should_checkpoint = (
            epoch % args.checkpoint_every == 0
            or epoch == args.epochs
        )
        if rank == 0 and should_checkpoint:
            resume_payload = _make_checkpoint(
                model=model,
                config=config,
                state_mean=state_mean,
                state_std=state_std,
                log_dt_mean=log_dt_mean,
                log_dt_std=log_dt_std,
                gas=gas,
                train_source=args.train_source,
                validation_source=args.validation_source,
                mechanism=args.mechanism,
                history=history,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                world_size=world_size,
                batch_size_per_device=args.batch_size_per_device,
                learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
                normalization_source=args.normalization_source,
                normalization_provenance=normalization_provenance,
            )
            resume_payload.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
            )
            _atomic_save(resume_payload, resume_output)

    if rank == 0:
        final_checkpoint = _make_checkpoint(
            model=model,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
            log_dt_mean=log_dt_mean,
            log_dt_std=log_dt_std,
            gas=gas,
            train_source=args.train_source,
            validation_source=args.validation_source,
            mechanism=args.mechanism,
            history=history,
            epoch=args.epochs,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            world_size=world_size,
            batch_size_per_device=args.batch_size_per_device,
            learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            normalization_source=args.normalization_source,
            normalization_provenance=normalization_provenance,
        )
        _atomic_save(final_checkpoint, final_output)
        summary = {
            "event": "training_complete",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint": str(output),
            "final_checkpoint": str(final_output),
            "resume_checkpoint": str(resume_output),
            "normalization_source": args.normalization_source,
            "normalization_provenance": normalization_provenance,
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
