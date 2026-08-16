from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from dfode_kit.data.cfd_conditioned import canonical_species_name
from dfode_kit.data.fluent_sensitivity import (
    FluentDIJVPBatch,
    deterministic_jvp_split_indices,
    deterministic_paired_batch_indices,
    load_fluent_di_jvp_dataset,
)
from dfode_kit.data.interval_pairs import load_interval_pair_arrays, load_interval_thermo_arrays
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
    THERMOCHEMICAL_HEAD_INPUT_STATE_TIME,
    THERMOCHEMICAL_OUTPUT_MODES,
    THERMOCHEMICAL_PROCESS_FEATURE_MODES,
    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
    THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS,
    THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER,
    NeuralPatankarIntervalModel,
    ReactionTrajectoryFreeEnergyModel,
)
from dfode_kit.physics.atom_conservation import reaction_affinity_over_rt, reaction_stoichiometry
from dfode_kit.training.fluent_deployment_losses import (
    FLUENT_NATIVE_DI_BACKEND,
    FluentDeploymentLossConfig,
    FluentJVPConsistencyConfig,
    LossTerm,
    fluent_deployment_losses,
    fluent_di_jvp_consistency_losses,
)


_FLUENT_NATIVE_DI_ALIASES = frozenset(
    {
        FLUENT_NATIVE_DI_BACKEND,
        "fluent-native-direct-integration",
    }
)
_DEPLOYMENT_TERM_NAMES = (
    "source",
    "mixture_molecular_weight",
    "density_increment",
)
_JVP_VALIDATION_FRACTION = 0.2


@dataclass
class PositiveIntervalTrainingConfig:
    variant: str = "neural-patankar"
    epochs: int = 300
    batch_size: int = 512
    learning_rate: float = 1e-3
    hidden_dim: int = 128
    latent_dim: int = 32
    transform_alpha: float = 0.1
    tp_loss_weight: float = 0.0
    state_loss_weight: float = 0.0
    delta_loss_weight: float = 1.0
    relative_delta_loss_weight: float = 1.0
    row_tail_relative_delta_loss_weight: float = 0.0
    row_tail_relative_delta_fraction: float = 0.1
    row_tail_relative_delta_floor: float = 1.0e-8
    row_tail_relative_delta_selection_weight: float = 0.0
    formation_energy_loss_weight: float = 0.0
    source_loss_weight: float = 0.0
    source_loss_scale: float = 1.0
    mixture_molecular_weight_loss_weight: float = 0.0
    mixture_molecular_weight_loss_scale: float = 1.0
    density_increment_loss_weight: float = 0.0
    density_increment_loss_scale: float = 1.0
    jvp_dataset: str | None = None
    source_jvp_loss_weight: float = 0.0
    source_jvp_loss_scale: float = 1.0
    thermochemical_output_mode: str = DELTA_TEMPERATURE_MODE
    thermochemical_head_input_mode: str = THERMOCHEMICAL_HEAD_INPUT_LEGACY
    thermochemical_head_hidden_dim: int = 0
    thermochemical_head_depth: int = 1
    total_enthalpy_head_hidden_dim: int = 0
    total_enthalpy_head_depth: int = 0
    thermochemical_aux_temperature: bool = False
    thermochemical_aux_temperature_loss_weight: float = 0.0
    thermochemical_delta_y_feature_mode: str = (
        THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY
    )
    thermochemical_delta_y_scale_floor: float = 1.0e-15
    thermochemical_delta_y_scales: tuple[float, ...] | None = None
    thermochemical_process_feature_mode: str = (
        THERMOCHEMICAL_PROCESS_FEATURE_NONE
    )
    thermochemical_state_high_low: bool = False
    thermochemical_state_low_scale: float = float(2**24)
    temperature_delta_scale: float = 10.0
    total_enthalpy_delta_scale: float = 1.0e3
    total_enthalpy_delta_loss_weight: float = 1.0
    total_enthalpy_target_transform: str = TOTAL_ENTHALPY_TARGET_IDENTITY
    total_enthalpy_transform_alpha: float = 0.1
    total_enthalpy_transformed_loss_weight: float = 0.0
    total_enthalpy_residual_mode: str = TOTAL_ENTHALPY_RESIDUAL_NONE
    total_enthalpy_residual_alpha: float = 0.1
    total_enthalpy_residual_transformed_loss_weight: float = 0.0
    total_enthalpy_sign_balanced_loss_weight: float = 0.0
    total_enthalpy_negative_abs_target_sum: float | None = None
    total_enthalpy_positive_abs_target_sum: float | None = None
    total_enthalpy_negative_count: int = 0
    total_enthalpy_positive_count: int = 0
    total_enthalpy_training_sample_count: int = 0
    temperature_rmse_weight: float = 0.0
    temperature_rmse_selection_weight: float = 0.0
    direction_loss_weight: float = 0.25
    opposing_extent_loss_weight: float = 0.0
    extent_scale: float = 1e-4
    demand_bias_init: float = -8.0
    availability_mode: str = "hard"
    availability_p_norm: float = 32.0
    trajectory_scale: float = 1e-5
    mobility_scale: float = 1e-4
    proximal_steps: int = 3
    proximal_beta: float = 1.0
    positivity_floor: float = 1e-30
    seed: int = 260624
    deterministic: bool = True
    log_every: int = 10


def _decode_attribute(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deployment_losses_enabled(config: PositiveIntervalTrainingConfig) -> bool:
    return any(
        weight > 0.0
        for weight in (
            config.source_loss_weight,
            config.mixture_molecular_weight_loss_weight,
            config.density_increment_loss_weight,
        )
    )


def _validate_jvp_configuration(config: PositiveIntervalTrainingConfig) -> None:
    if not np.isfinite(config.source_jvp_loss_weight) or config.source_jvp_loss_weight < 0.0:
        raise ValueError("source_jvp_loss_weight must be finite and nonnegative")
    if not np.isfinite(config.source_jvp_loss_scale) or config.source_jvp_loss_scale <= 0.0:
        raise ValueError("source_jvp_loss_scale must be finite and positive")
    if config.source_jvp_loss_weight > 0.0 and not config.jvp_dataset:
        raise ValueError("--jvp-dataset is required when source JVP loss is weighted")


def _validate_row_tail_configuration(
    config: PositiveIntervalTrainingConfig,
) -> None:
    if (
        not np.isfinite(config.row_tail_relative_delta_loss_weight)
        or config.row_tail_relative_delta_loss_weight < 0.0
    ):
        raise ValueError(
            "row_tail_relative_delta_loss_weight must be finite and nonnegative"
        )
    if (
        not np.isfinite(config.row_tail_relative_delta_fraction)
        or not 0.0 < config.row_tail_relative_delta_fraction <= 1.0
    ):
        raise ValueError(
            "row_tail_relative_delta_fraction must be finite and in (0, 1]"
        )
    if (
        not np.isfinite(config.row_tail_relative_delta_floor)
        or config.row_tail_relative_delta_floor <= 0.0
    ):
        raise ValueError(
            "row_tail_relative_delta_floor must be finite and positive"
        )
    if (
        not np.isfinite(config.row_tail_relative_delta_selection_weight)
        or config.row_tail_relative_delta_selection_weight < 0.0
    ):
        raise ValueError(
            "row_tail_relative_delta_selection_weight must be finite and "
            "nonnegative"
        )


def _canonical_fluent_label_backend(attributes: dict) -> str | None:
    backend = _decode_attribute(attributes.get("label_backend"))
    if backend in _FLUENT_NATIVE_DI_ALIASES:
        return FLUENT_NATIVE_DI_BACKEND
    if backend in {
        "cantera-cvode",
        "cantera-cvode-bridge",
        "dfode-cantera-cvode-bridge-v1",
    }:
        return "cantera-cvode-bridge"
    return None


def _require_deployment_label_sources(
    config: PositiveIntervalTrainingConfig,
    train_attributes: dict,
    validation_attributes: dict,
) -> tuple[str | None, str | None]:
    train_backend = _canonical_fluent_label_backend(train_attributes)
    validation_backend = _canonical_fluent_label_backend(validation_attributes)
    requires_deployment_labels = (
        _deployment_losses_enabled(config)
        or config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
        or bool(config.jvp_dataset)
        or config.source_jvp_loss_weight > 0.0
    )
    if requires_deployment_labels and (
        train_backend is None
        or validation_backend is None
        or train_backend != validation_backend
    ):
        raise ValueError(
            "deployment-oriented losses require train and validation labels "
            "from one identical declared backend contract; supported contracts "
            "are Fluent native direct integration and the DFODE Cantera/CVODE "
            "bridge"
        )
    return train_backend, validation_backend


@dataclass(frozen=True)
class _JVPTrainingData:
    batch: FluentDIJVPBatch
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _prepare_jvp_training_data(
    config: PositiveIntervalTrainingConfig,
    *,
    species_names,
    train_dt,
    validation_dt,
) -> _JVPTrainingData | None:
    _validate_jvp_configuration(config)
    if not config.jvp_dataset:
        return None
    batch = load_fluent_di_jvp_dataset(config.jvp_dataset)
    expected_species = tuple(canonical_species_name(name) for name in species_names)
    observed_species = tuple(
        canonical_species_name(name) for name in batch.species_names
    )
    if observed_species != expected_species:
        raise ValueError(
            "JVP dataset species order must exactly match the mechanism species order"
        )
    exact_dt = float(batch.provenance["exact_dt_seconds"])
    for name, values in (
        ("training", train_dt),
        ("validation", validation_dt),
    ):
        intervals = np.asarray(values, dtype=np.float64).reshape(-1)
        if intervals.size == 0 or not np.all(intervals == exact_dt):
            raise ValueError(
                f"{name} endpoint dt must exactly match JVP exact_dt_seconds "
                f"({exact_dt:.17g} s)"
            )
    train_indices, validation_indices = deterministic_jvp_split_indices(
        batch.x.shape[0],
        validation_fraction=_JVP_VALIDATION_FRACTION,
        seed=config.seed,
    )
    return _JVPTrainingData(batch, train_indices, validation_indices)


def _jvp_loss_config(
    config: PositiveIntervalTrainingConfig,
) -> FluentJVPConsistencyConfig:
    return FluentJVPConsistencyConfig(
        endpoint=LossTerm(0.0, 1.0),
        source=LossTerm(
            config.source_jvp_loss_weight,
            config.source_jvp_loss_scale,
        ),
    )


def _validate_thermochemical_configuration(
    config: PositiveIntervalTrainingConfig,
) -> None:
    if config.thermochemical_output_mode not in THERMOCHEMICAL_OUTPUT_MODES:
        raise ValueError(
            "thermochemical_output_mode must be one of "
            f"{THERMOCHEMICAL_OUTPUT_MODES}"
        )
    if config.total_enthalpy_head_hidden_dim < 0:
        raise ValueError(
            "total_enthalpy_head_hidden_dim must be nonnegative"
        )
    if config.total_enthalpy_head_depth < 0:
        raise ValueError("total_enthalpy_head_depth must be nonnegative")
    if (
        (
            config.total_enthalpy_head_hidden_dim
            or config.total_enthalpy_head_depth
        )
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "total-enthalpy head overrides require delta-h-total mode"
        )
    if (
        not np.isfinite(config.total_enthalpy_delta_scale)
        or config.total_enthalpy_delta_scale <= 0.0
    ):
        raise ValueError("total_enthalpy_delta_scale must be finite and positive")
    if config.total_enthalpy_delta_loss_weight < 0.0:
        raise ValueError("total_enthalpy_delta_loss_weight must be nonnegative")
    if config.total_enthalpy_target_transform not in (
        TOTAL_ENTHALPY_TARGET_TRANSFORMS
    ):
        raise ValueError(
            "total_enthalpy_target_transform must be one of "
            f"{TOTAL_ENTHALPY_TARGET_TRANSFORMS}"
        )
    if (
        not np.isfinite(config.total_enthalpy_transform_alpha)
        or not 0.0 < config.total_enthalpy_transform_alpha <= 1.0
    ):
        raise ValueError(
            "total_enthalpy_transform_alpha must be finite and in (0, 1]"
        )
    if (
        not np.isfinite(config.total_enthalpy_transformed_loss_weight)
        or config.total_enthalpy_transformed_loss_weight < 0.0
    ):
        raise ValueError(
            "total_enthalpy_transformed_loss_weight must be finite and "
            "nonnegative"
        )
    if (
        config.total_enthalpy_target_transform
        != TOTAL_ENTHALPY_TARGET_IDENTITY
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "transformed total-enthalpy targets require delta-h-total mode"
        )
    if (
        config.total_enthalpy_target_transform
        == TOTAL_ENTHALPY_TARGET_SIGNED_POWER
        and config.total_enthalpy_transformed_loss_weight <= 0.0
    ):
        raise ValueError(
            "signed-power total-enthalpy targets require a positive "
            "transformed loss weight"
        )
    if config.total_enthalpy_residual_mode not in TOTAL_ENTHALPY_RESIDUAL_MODES:
        raise ValueError(
            "total_enthalpy_residual_mode must be one of "
            f"{TOTAL_ENTHALPY_RESIDUAL_MODES}"
        )
    if (
        not np.isfinite(config.total_enthalpy_residual_alpha)
        or not 0.0 < config.total_enthalpy_residual_alpha <= 1.0
    ):
        raise ValueError(
            "total_enthalpy_residual_alpha must be finite and in (0, 1]"
        )
    if (
        not np.isfinite(
            config.total_enthalpy_residual_transformed_loss_weight
        )
        or config.total_enthalpy_residual_transformed_loss_weight < 0.0
    ):
        raise ValueError(
            "total_enthalpy_residual_transformed_loss_weight must be finite "
            "and nonnegative"
        )
    if config.total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE:
        if config.thermochemical_output_mode != DELTA_H_TOTAL_MODE:
            raise ValueError(
                "total-enthalpy residual correction requires delta-h-total mode"
            )
        if (
            config.total_enthalpy_target_transform
            != TOTAL_ENTHALPY_TARGET_IDENTITY
        ):
            raise ValueError(
                "total-enthalpy residual correction requires an identity "
                "physical baseline head"
            )
        if config.total_enthalpy_transformed_loss_weight != 0.0:
            raise ValueError(
                "absolute-target transformed loss must be zero in residual mode"
            )
        if config.total_enthalpy_residual_transformed_loss_weight <= 0.0:
            raise ValueError(
                "signed-power residual correction requires a positive "
                "residual transformed loss weight"
            )
    if (
        not np.isfinite(config.total_enthalpy_sign_balanced_loss_weight)
        or config.total_enthalpy_sign_balanced_loss_weight < 0.0
    ):
        raise ValueError(
            "total_enthalpy_sign_balanced_loss_weight must be finite and "
            "nonnegative"
        )
    if (
        config.total_enthalpy_sign_balanced_loss_weight > 0.0
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "sign-balanced total-enthalpy loss requires delta-h-total mode"
        )
    if config.thermochemical_aux_temperature_loss_weight < 0.0:
        raise ValueError(
            "thermochemical_aux_temperature_loss_weight must be nonnegative"
        )
    if (
        config.thermochemical_aux_temperature_loss_weight > 0.0
        and not config.thermochemical_aux_temperature
    ):
        raise ValueError(
            "thermochemical auxiliary temperature loss requires the "
            "auxiliary temperature head"
        )
    if config.thermochemical_delta_y_feature_mode not in (
        THERMOCHEMICAL_DELTA_Y_FEATURE_MODES
    ):
        raise ValueError(
            "thermochemical_delta_y_feature_mode must be one of "
            f"{THERMOCHEMICAL_DELTA_Y_FEATURE_MODES}"
        )
    if (
        not np.isfinite(config.thermochemical_delta_y_scale_floor)
        or config.thermochemical_delta_y_scale_floor <= 0.0
    ):
        raise ValueError(
            "thermochemical_delta_y_scale_floor must be finite and positive"
        )
    if config.thermochemical_delta_y_scales is not None:
        delta_y_scales = np.asarray(
            config.thermochemical_delta_y_scales, dtype=np.float64
        )
        if delta_y_scales.ndim != 1:
            raise ValueError(
                "thermochemical_delta_y_scales must be one-dimensional"
            )
        if not np.all(np.isfinite(delta_y_scales)) or np.any(
            delta_y_scales <= 0.0
        ):
            raise ValueError(
                "thermochemical_delta_y_scales must be finite and positive"
            )
    if (
        config.thermochemical_delta_y_feature_mode
        == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "asinh thermochemical delta-Y features require delta-h-total mode"
        )
    if config.thermochemical_process_feature_mode not in (
        THERMOCHEMICAL_PROCESS_FEATURE_MODES
    ):
        raise ValueError(
            "thermochemical_process_feature_mode must be one of "
            f"{THERMOCHEMICAL_PROCESS_FEATURE_MODES}"
        )
    if (
        config.thermochemical_process_feature_mode
        != THERMOCHEMICAL_PROCESS_FEATURE_NONE
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "thermochemical process features require delta-h-total mode"
        )
    if (
        config.thermochemical_aux_temperature
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "thermochemical auxiliary temperature requires delta-h-total mode"
        )
    if (
        not np.isfinite(config.thermochemical_state_low_scale)
        or config.thermochemical_state_low_scale <= 0.0
    ):
        raise ValueError(
            "thermochemical_state_low_scale must be finite and positive"
        )
    if (
        config.thermochemical_state_high_low
        and config.thermochemical_output_mode != DELTA_H_TOTAL_MODE
    ):
        raise ValueError(
            "thermochemical high/low state features require delta-h-total mode"
        )
    if (
        config.thermochemical_state_high_low
        and config.thermochemical_head_input_mode
        != THERMOCHEMICAL_HEAD_INPUT_STATE_TIME
    ):
        raise ValueError(
            "thermochemical high/low state features require the state-time "
            "thermochemical head input mode"
        )
    if config.thermochemical_output_mode != DELTA_H_TOTAL_MODE:
        return
    if config.variant != "neural-patankar":
        raise ValueError(
            "delta-h-total thermochemical output currently requires "
            "variant='neural-patankar'"
        )
    if config.tp_loss_weight != 0.0 or config.temperature_rmse_weight != 0.0:
        raise ValueError(
            "delta-h-total mode has no independent temperature prediction; "
            "temperature/TP loss weights must be zero"
        )
    if config.density_increment_loss_weight != 0.0:
        raise ValueError(
            "density-increment loss is unavailable in delta-h-total mode until "
            "Fluent thermodynamic temperature inversion is part of training"
        )


def _load_total_enthalpy_increment_labels(
    path: str,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = ("h_total_before", "h_total_after")
    with h5py.File(path, "r") as h5:
        if "pairs" not in h5:
            raise ValueError(
                f"{path} is missing the 'pairs' group required for Fluent "
                "total-enthalpy labels"
            )
        pairs = h5["pairs"]
        missing = [name for name in required if name not in pairs]
        if missing:
            raise ValueError(
                f"{path} is missing required Fluent total-enthalpy label "
                f"dataset(s): {', '.join('pairs/' + name for name in missing)}"
            )
        before = np.asarray(pairs["h_total_before"][:], dtype=np.float64).reshape(-1)
        after = np.asarray(pairs["h_total_after"][:], dtype=np.float64).reshape(-1)
    if before.size != expected_rows or after.size != expected_rows:
        raise ValueError(
            f"{path} total-enthalpy labels have {before.size}/{after.size} rows; "
            f"expected {expected_rows}"
        )
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError(f"{path} total-enthalpy labels contain non-finite values")
    return before, after, after - before


def compute_total_enthalpy_sign_statistics(target_delta_h) -> dict[str, float | int]:
    """Compute fixed sign denominators from the complete training split."""
    target = np.asarray(target_delta_h, dtype=np.float64).reshape(-1)
    if target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("Total-enthalpy targets must be finite and nonempty")
    negative = target < 0.0
    positive = target > 0.0
    negative_count = int(np.count_nonzero(negative))
    positive_count = int(np.count_nonzero(positive))
    negative_abs_target_sum = float(
        np.sum(np.abs(target[negative]), dtype=np.float64)
    )
    positive_abs_target_sum = float(
        np.sum(np.abs(target[positive]), dtype=np.float64)
    )
    if (
        negative_count == 0
        or positive_count == 0
        or negative_abs_target_sum <= 0.0
        or positive_abs_target_sum <= 0.0
    ):
        raise ValueError(
            "Sign-balanced total-enthalpy loss requires nonempty negative "
            "and positive target strata"
        )
    return {
        "training_sample_count": int(target.size),
        "negative_count": negative_count,
        "positive_count": positive_count,
        "zero_count": int(target.size - negative_count - positive_count),
        "negative_abs_target_sum": negative_abs_target_sum,
        "positive_abs_target_sum": positive_abs_target_sum,
        "negative_mean_abs_target": negative_abs_target_sum / negative_count,
        "positive_mean_abs_target": positive_abs_target_sum / positive_count,
    }


def total_enthalpy_sign_balanced_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    negative_abs_target_sum: float,
    positive_abs_target_sum: float,
    training_sample_count: int,
) -> torch.Tensor:
    """Estimate equal negative/positive stratum NMAE additively per row.

    Fixed full-training denominators make DDP mean-gradient reduction
    equivalent to the global objective for equal local batch sizes. Exact-zero
    targets receive no auxiliary weight and remain governed by raw MAE.
    """
    predicted = prediction.reshape(-1)
    observed = target.reshape(-1).to(predicted.dtype)
    if predicted.shape != observed.shape:
        raise ValueError("Total-enthalpy prediction and target shapes differ")
    if (
        not np.isfinite(negative_abs_target_sum)
        or negative_abs_target_sum <= 0.0
        or not np.isfinite(positive_abs_target_sum)
        or positive_abs_target_sum <= 0.0
        or training_sample_count <= 0
    ):
        raise ValueError("Sign-balanced training statistics must be positive")
    negative_denominator = torch.as_tensor(
        negative_abs_target_sum,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    positive_denominator = torch.as_tensor(
        positive_abs_target_sum,
        dtype=predicted.dtype,
        device=predicted.device,
    )
    row_weight = torch.where(
        observed < 0.0,
        1.0 / negative_denominator,
        torch.where(
            observed > 0.0,
            1.0 / positive_denominator,
            torch.zeros((), dtype=predicted.dtype, device=predicted.device),
        ),
    )
    return (
        0.5
        * float(training_sample_count)
        * torch.mean(torch.abs(predicted - observed) * row_weight)
    )


def _deployment_loss_config(
    config: PositiveIntervalTrainingConfig,
) -> FluentDeploymentLossConfig:
    return FluentDeploymentLossConfig(
        source=LossTerm(config.source_loss_weight, config.source_loss_scale),
        mixture_molecular_weight=LossTerm(
            config.mixture_molecular_weight_loss_weight,
            config.mixture_molecular_weight_loss_scale,
        ),
        density_increment=LossTerm(
            config.density_increment_loss_weight,
            config.density_increment_loss_scale,
        ),
        enthalpy=LossTerm(0.0, 1.0),
        heat_release=LossTerm(0.0, 1.0),
    )


def _deployment_loss_for_batch(
    *,
    loss_config: FluentDeploymentLossConfig,
    current_y: torch.Tensor,
    predicted_delta_y: torch.Tensor,
    target_delta_y: torch.Tensor,
    current_temperature: torch.Tensor,
    predicted_temperature: torch.Tensor,
    target_temperature: torch.Tensor,
    pressure: torch.Tensor,
    dt: torch.Tensor,
    molecular_weights: torch.Tensor,
    zero_species_enthalpy: torch.Tensor,
) -> dict[str, object]:
    # The endpoint head is unconstrained at initialization. Use a smooth lower
    # bound only inside physical density evaluation so the first optimization
    # batches remain defined. Above ordinary combustion temperatures this is
    # numerically the identity; the raw head remains the exported prediction.
    density_temperature_floor = torch.as_tensor(
        1.0,
        dtype=predicted_temperature.dtype,
        device=predicted_temperature.device,
    )
    density_temperature_width = torch.as_tensor(
        10.0,
        dtype=predicted_temperature.dtype,
        device=predicted_temperature.device,
    )
    predicted_temperature_for_density = density_temperature_floor + (
        density_temperature_width
        * F.softplus(
            (predicted_temperature - density_temperature_floor)
            / density_temperature_width
        )
    )
    return fluent_deployment_losses(
        current_y=current_y,
        predicted_delta_y=predicted_delta_y,
        target_delta_y=target_delta_y,
        current_temperature=current_temperature,
        predicted_temperature=predicted_temperature_for_density,
        target_temperature=target_temperature,
        pressure=pressure,
        dt=dt,
        molecular_weights=molecular_weights,
        species_enthalpy=lambda _temperature: zero_species_enthalpy,
        config=loss_config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )


def _deployment_endpoint_temperature(
    output: dict[str, torch.Tensor],
    prediction: torch.Tensor,
    current_temperature: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> torch.Tensor:
    if "delta_temperature" in output:
        return current_temperature + output["delta_temperature"].reshape(-1)
    return prediction[:, 0] * state_std[0] + state_mean[0]


def _empty_loss_accumulator() -> dict[str, dict[str, float]]:
    return {
        name: {"raw_mae": 0.0, "scaled": 0.0, "weighted": 0.0}
        for name in _DEPLOYMENT_TERM_NAMES
    }


def _accumulate_loss_report(accumulator, report, count: int) -> None:
    for name in _DEPLOYMENT_TERM_NAMES:
        term = report["terms"][name]
        accumulator[name]["raw_mae"] += float(term["raw_mae"].detach()) * count
        accumulator[name]["scaled"] += float(term["scaled"].detach()) * count
        accumulator[name]["weighted"] += float(term["contribution"].detach()) * count


def _flatten_loss_report(prefix, accumulator, count, loss_config) -> dict[str, float]:
    result = {}
    for name in _DEPLOYMENT_TERM_NAMES:
        term_config = getattr(loss_config, name)
        result[f"{prefix}_{name}_raw_mae"] = accumulator[name]["raw_mae"] / count
        result[f"{prefix}_{name}_scaled"] = accumulator[name]["scaled"] / count
        result[f"{prefix}_{name}_weighted"] = accumulator[name]["weighted"] / count
        result[f"{prefix}_{name}_scale"] = float(term_config.scale)
        result[f"{prefix}_{name}_weight"] = float(term_config.weight)
    return result


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
        return _affinity_features_from_states(current, gas)


def _affinity_features_from_states(current: np.ndarray, gas) -> np.ndarray:
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


def compute_thermochemical_delta_y_scales(
    current_state: np.ndarray,
    target_state: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """Compute deterministic per-species scales from training endpoints only."""
    current = np.asarray(current_state, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    if current.shape != target.shape or current.ndim != 2:
        raise ValueError("current and target training states must have equal 2D shapes")
    if current.shape[1] < 3:
        raise ValueError("training states must contain T, P, and species columns")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("thermochemical delta-Y scale floor must be positive")
    scales = np.empty(current.shape[1] - 2, dtype=np.float64)
    for species_index in range(scales.size):
        absolute_delta = np.abs(
            target[:, species_index + 2]
            - current[:, species_index + 2]
        )
        scales[species_index] = max(
            float(np.quantile(absolute_delta, 0.90, method="linear")),
            float(floor),
        )
    return scales


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
            demand_bias_init=config.demand_bias_init,
            availability_floor=config.positivity_floor,
            temperature_delta_scale=config.temperature_delta_scale,
            thermochemical_output_mode=config.thermochemical_output_mode,
            total_enthalpy_delta_scale=config.total_enthalpy_delta_scale,
            total_enthalpy_target_transform=(
                config.total_enthalpy_target_transform
            ),
            total_enthalpy_transform_alpha=(
                config.total_enthalpy_transform_alpha
            ),
            total_enthalpy_residual_mode=(
                config.total_enthalpy_residual_mode
            ),
            total_enthalpy_residual_alpha=(
                config.total_enthalpy_residual_alpha
            ),
            thermochemical_head_input_mode=(
                config.thermochemical_head_input_mode
            ),
            thermochemical_head_hidden_dim=(
                config.thermochemical_head_hidden_dim or None
            ),
            thermochemical_head_depth=config.thermochemical_head_depth,
            total_enthalpy_head_hidden_dim=(
                config.total_enthalpy_head_hidden_dim
            ),
            total_enthalpy_head_depth=config.total_enthalpy_head_depth,
            thermochemical_aux_temperature=(
                config.thermochemical_aux_temperature
            ),
            thermochemical_delta_y_feature_mode=(
                config.thermochemical_delta_y_feature_mode
            ),
            thermochemical_delta_y_scales=(
                None
                if config.thermochemical_delta_y_scales is None
                else np.asarray(
                    config.thermochemical_delta_y_scales,
                    dtype=np.float64,
                )
            ),
            thermochemical_process_feature_mode=(
                config.thermochemical_process_feature_mode
            ),
            thermochemical_state_high_low=(
                config.thermochemical_state_high_low
            ),
            thermochemical_state_low_scale=(
                config.thermochemical_state_low_scale
            ),
            availability_mode=config.availability_mode,
            availability_p_norm=config.availability_p_norm,
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


def _distributed_rank_and_world_size() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _predict_jvp_endpoint(
    model,
    state: torch.Tensor,
    dt: torch.Tensor,
    *,
    config: PositiveIntervalTrainingConfig,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    log_dt_mean: torch.Tensor,
    log_dt_std: torch.Tensor,
    affinity: torch.Tensor | None,
) -> torch.Tensor:
    normalized_state_float64 = (state - state_mean) / state_std
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
        (torch.log(dt.clamp_min(1.0e-300))[:, None] - log_dt_mean)
        / log_dt_std
    ).to(torch.float32)
    current_species = state[:, 2:]
    if config.variant == "reaction-trajectory":
        if affinity is None:
            raise ValueError("reaction-trajectory JVP inference requires affinity")
        output = model(
            normalized_state,
            normalized_log_dt,
            affinity,
            current_species=current_species,
        )
    else:
        output = model(
            normalized_state,
            normalized_log_dt,
            current_species=current_species,
            thermochemical_state_low=thermochemical_state_low,
        )
    # Only the species block enters g=(Phi_Y-Y)/dt. Retaining input T,p here
    # avoids inventing an endpoint TP contract for delta-h-total models.
    return torch.cat((state[:, :2], output["next_species"]), dim=1)


def _jvp_loss_for_indices(
    model,
    jvp_data: _JVPTrainingData,
    indices,
    *,
    config: PositiveIntervalTrainingConfig,
    loss_config: FluentJVPConsistencyConfig,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    log_dt_mean: torch.Tensor,
    log_dt_std: torch.Tensor,
    device: torch.device,
    base_affinity: np.ndarray | None,
    perturbed_affinity: np.ndarray | None,
) -> dict[str, object]:
    pair_ids = np.asarray(indices, dtype=np.int64)
    batch = jvp_data.batch
    current = torch.as_tensor(batch.x[pair_ids], dtype=torch.float64, device=device)
    perturbed = torch.as_tensor(
        batch.x_perturbed[pair_ids], dtype=torch.float64, device=device
    )
    interval = torch.as_tensor(batch.dt[pair_ids], dtype=torch.float64, device=device)
    epsilon = torch.as_tensor(
        batch.epsilon[pair_ids], dtype=torch.float64, device=device
    )
    affinity = (
        None
        if base_affinity is None
        else torch.as_tensor(base_affinity[pair_ids], dtype=torch.float32, device=device)
    )
    affinity_perturbed = (
        None
        if perturbed_affinity is None
        else torch.as_tensor(
            perturbed_affinity[pair_ids], dtype=torch.float32, device=device
        )
    )
    predicted = _predict_jvp_endpoint(
        model,
        current,
        interval,
        config=config,
        state_mean=state_mean,
        state_std=state_std,
        log_dt_mean=log_dt_mean,
        log_dt_std=log_dt_std,
        affinity=affinity,
    )
    predicted_perturbed = _predict_jvp_endpoint(
        model,
        perturbed,
        interval,
        config=config,
        state_mean=state_mean,
        state_std=state_std,
        log_dt_mean=log_dt_mean,
        log_dt_std=log_dt_std,
        affinity=affinity_perturbed,
    )
    return fluent_di_jvp_consistency_losses(
        current_state=current,
        perturbed_state=perturbed,
        predicted_endpoint=predicted,
        predicted_perturbed_endpoint=predicted_perturbed,
        target_di_endpoint=torch.as_tensor(
            batch.phi_di_x[pair_ids], dtype=torch.float64, device=device
        ),
        target_di_perturbed_endpoint=torch.as_tensor(
            batch.phi_di_x_perturbed[pair_ids],
            dtype=torch.float64,
            device=device,
        ),
        epsilon=epsilon,
        dt=interval,
        config=loss_config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )


def _empty_jvp_accumulator() -> dict[str, float]:
    return {
        "absolute_error_sum": 0.0,
        "target_absolute_sum": 0.0,
        "dot_sum": 0.0,
        "prediction_square_sum": 0.0,
        "target_square_sum": 0.0,
        "element_count": 0.0,
        "pair_count": 0.0,
    }


def _accumulate_jvp_report(
    accumulator: dict[str, float], report: dict[str, object]
) -> None:
    predicted = report["derived"]["predicted_source_jvp"].detach()
    target = report["derived"]["target_source_jvp"].detach()
    accumulator["absolute_error_sum"] += float(torch.sum(torch.abs(predicted - target)))
    accumulator["target_absolute_sum"] += float(torch.sum(torch.abs(target)))
    accumulator["dot_sum"] += float(torch.sum(predicted * target))
    accumulator["prediction_square_sum"] += float(torch.sum(predicted.square()))
    accumulator["target_square_sum"] += float(torch.sum(target.square()))
    accumulator["element_count"] += float(target.numel())
    accumulator["pair_count"] += float(target.shape[0])


def _reduce_jvp_accumulator(
    accumulator: dict[str, float], device: torch.device
) -> dict[str, float]:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return accumulator
    keys = tuple(accumulator)
    values = torch.tensor(
        [accumulator[key] for key in keys], dtype=torch.float64, device=device
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, values.cpu())}


def _finalize_jvp_metrics(
    prefix: str,
    accumulator: dict[str, float],
    config: PositiveIntervalTrainingConfig,
) -> dict[str, float]:
    element_count = max(accumulator["element_count"], 1.0)
    raw_mae = accumulator["absolute_error_sum"] / element_count
    target_l1 = accumulator["target_absolute_sum"]
    nmae = (
        accumulator["absolute_error_sum"] / target_l1
        if target_l1 > 0.0
        else (0.0 if accumulator["absolute_error_sum"] == 0.0 else float("inf"))
    )
    norm_product = (
        accumulator["prediction_square_sum"]
        * accumulator["target_square_sum"]
    ) ** 0.5
    cosine = (
        accumulator["dot_sum"] / norm_product
        if norm_product > 0.0
        else (1.0 if accumulator["absolute_error_sum"] == 0.0 else 0.0)
    )
    fixed_scale_loss = raw_mae / config.source_jvp_loss_scale
    return {
        f"{prefix}_source_jvp_raw_mae": raw_mae,
        f"{prefix}_source_jvp_fixed_scale_loss": fixed_scale_loss,
        f"{prefix}_source_jvp_weighted_loss": (
            config.source_jvp_loss_weight * fixed_scale_loss
        ),
        f"{prefix}_source_jvp_nmae": nmae,
        f"{prefix}_source_jvp_cosine": cosine,
        f"{prefix}_source_jvp_pairs": int(accumulator["pair_count"]),
    }


def _evaluate_jvp_partition(
    model,
    jvp_data: _JVPTrainingData,
    indices: np.ndarray,
    *,
    prefix: str,
    config: PositiveIntervalTrainingConfig,
    loss_config: FluentJVPConsistencyConfig,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    log_dt_mean: torch.Tensor,
    log_dt_std: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    base_affinity: np.ndarray | None,
    perturbed_affinity: np.ndarray | None,
) -> dict[str, float]:
    accumulator = _empty_jvp_accumulator()
    batches = deterministic_paired_batch_indices(
        indices,
        batch_size=config.batch_size,
        seed=config.seed,
        epoch=0,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )
    for pair_ids in batches:
        report = _jvp_loss_for_indices(
            model,
            jvp_data,
            pair_ids,
            config=config,
            loss_config=loss_config,
            state_mean=state_mean,
            state_std=state_std,
            log_dt_mean=log_dt_mean,
            log_dt_std=log_dt_std,
            device=device,
            base_affinity=base_affinity,
            perturbed_affinity=perturbed_affinity,
        )
        _accumulate_jvp_report(accumulator, report)
    accumulator = _reduce_jvp_accumulator(accumulator, device)
    return _finalize_jvp_metrics(prefix, accumulator, config)


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
    _validate_thermochemical_configuration(config)
    _validate_jvp_configuration(config)
    _set_seed(config.seed, config.deterministic)
    train_current, train_target, train_dt, *_rest = load_interval_pair_arrays(train_source, dtype=np.float64)
    val_current, val_target, val_dt, *_val_rest = load_interval_pair_arrays(validation_source, dtype=np.float64)
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
    train_attributes = _rest[-1]
    validation_attributes = _val_rest[-1]
    train_label_backend, validation_label_backend = _require_deployment_label_sources(
        config,
        train_attributes,
        validation_attributes,
    )
    train_delta_h_total = val_delta_h_total = None
    if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE:
        _train_h_before, _train_h_after, train_delta_h_total = (
            _load_total_enthalpy_increment_labels(
                train_source,
                expected_rows=train_current.shape[0],
            )
        )
        _val_h_before, _val_h_after, val_delta_h_total = (
            _load_total_enthalpy_increment_labels(
                validation_source,
                expected_rows=val_current.shape[0],
            )
        )
    deployment_loss_config = _deployment_loss_config(config)
    gas = ct.Solution(mech_path, phase_name) if phase_name else ct.Solution(mech_path)
    if train_current.shape[1] != gas.n_species + 2:
        raise ValueError("dataset species dimension does not match mechanism")
    jvp_training_data = _prepare_jvp_training_data(
        config,
        species_names=gas.species_names,
        train_dt=train_dt,
        validation_dt=val_dt,
    )
    jvp_loss_config = _jvp_loss_config(config)

    state_mean = np.mean(train_current, axis=0)
    state_std = np.std(train_current, axis=0)
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    train_log_dt = np.log(np.maximum(train_dt, 1e-300))[:, None]
    val_log_dt = np.log(np.maximum(val_dt, 1e-300))[:, None]
    log_dt_mean = np.mean(train_log_dt, axis=0)
    log_dt_std = np.std(train_log_dt, axis=0)
    log_dt_std = np.where(log_dt_std > 1.0e-8, log_dt_std, 1.0)

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, gas, train_current.shape[1]).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_normalized_float64 = (train_current - state_mean) / state_std
    train_x_numpy = train_normalized_float64.astype(np.float32)
    train_x = torch.tensor(
        train_x_numpy, dtype=torch.float32, device=torch_device
    )
    train_x_low = None
    if config.thermochemical_state_high_low:
        train_x_low = torch.tensor(
            (
                (
                    train_normalized_float64
                    - train_x_numpy.astype(np.float64)
                )
                * config.thermochemical_state_low_scale
            ).astype(np.float32),
            dtype=torch.float32,
            device=torch_device,
        )
    train_t = torch.tensor((train_log_dt - log_dt_mean) / log_dt_std, dtype=torch.float32, device=torch_device)
    train_y0 = torch.tensor(train_current[:, 2:], dtype=torch.float64, device=torch_device)
    train_true = torch.tensor(train_target, dtype=torch.float64, device=torch_device)
    train_current_tp = torch.tensor(train_current[:, :2], dtype=torch.float64, device=torch_device)
    train_dt_physical = torch.tensor(train_dt, dtype=torch.float64, device=torch_device)
    val_normalized_float64 = (val_current - state_mean) / state_std
    val_x_numpy = val_normalized_float64.astype(np.float32)
    val_x = torch.tensor(
        val_x_numpy, dtype=torch.float32, device=torch_device
    )
    val_x_low = None
    if config.thermochemical_state_high_low:
        val_x_low = torch.tensor(
            (
                (
                    val_normalized_float64
                    - val_x_numpy.astype(np.float64)
                )
                * config.thermochemical_state_low_scale
            ).astype(np.float32),
            dtype=torch.float32,
            device=torch_device,
        )
    val_t = torch.tensor((val_log_dt - log_dt_mean) / log_dt_std, dtype=torch.float32, device=torch_device)
    val_y0 = torch.tensor(val_current[:, 2:], dtype=torch.float64, device=torch_device)
    val_true = torch.tensor(val_target, dtype=torch.float64, device=torch_device)
    val_current_tp = torch.tensor(val_current[:, :2], dtype=torch.float64, device=torch_device)
    val_dt_physical = torch.tensor(val_dt, dtype=torch.float64, device=torch_device)
    train_delta_h_total_t = val_delta_h_total_t = None
    if train_delta_h_total is not None:
        train_delta_h_total_t = torch.tensor(
            train_delta_h_total,
            dtype=torch.float64,
            device=torch_device,
        )
        val_delta_h_total_t = torch.tensor(
            val_delta_h_total,
            dtype=torch.float64,
            device=torch_device,
        )

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
    log_dt_mean_t = torch.tensor(log_dt_mean, dtype=torch.float64, device=torch_device)
    log_dt_std_t = torch.tensor(log_dt_std, dtype=torch.float64, device=torch_device)
    molecular_weights_t = torch.tensor(
        gas.molecular_weights,
        dtype=torch.float64,
        device=torch_device,
    )
    zero_species_enthalpy = torch.zeros(
        gas.n_species,
        dtype=torch.float64,
        device=torch_device,
    )
    jvp_base_affinity = jvp_perturbed_affinity = None
    if jvp_training_data is not None and config.variant == "reaction-trajectory":
        jvp_base_affinity = _affinity_features_from_states(
            jvp_training_data.batch.x, gas
        )
        jvp_perturbed_affinity = _affinity_features_from_states(
            jvp_training_data.batch.x_perturbed, gas
        )
    jvp_rank, jvp_world_size = _distributed_rank_and_world_size()

    def loss_for(indices, training):
        x = train_x[indices] if training else val_x[indices]
        x_low = (
            None
            if train_x_low is None
            else (train_x_low[indices] if training else val_x_low[indices])
        )
        t = train_t[indices] if training else val_t[indices]
        y0 = train_y0[indices] if training else val_y0[indices]
        truth = train_true[indices] if training else val_true[indices]
        current_tp = train_current_tp[indices] if training else val_current_tp[indices]
        dt_physical = train_dt_physical[indices] if training else val_dt_physical[indices]
        if config.variant == "reaction-trajectory":
            affinity = train_affinity[indices] if training else val_affinity[indices]
            output = model(x, t, affinity, current_species=y0)
        else:
            output = model(
                x,
                t,
                current_species=y0,
                thermochemical_state_low=x_low,
            )
        prediction = output["next_state"]
        pred_y = output["next_species"]
        true_y = truth[:, 2:]
        if config.thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
            true_tp_norm = (truth[:, :2] - state_mean_t[:2]) / state_std_t[:2]
            tp_loss = F.l1_loss(prediction[:, :2], true_tp_norm)
            delta_h_total_loss = pred_y.sum() * 0.0
            delta_h_total_raw_mae = pred_y.sum() * 0.0
            delta_h_total_transformed_loss = pred_y.sum() * 0.0
            delta_h_total_residual_transformed_loss = pred_y.sum() * 0.0
        else:
            tp_loss = pred_y.sum() * 0.0
            target_delta_h_total = (
                train_delta_h_total_t[indices]
                if training
                else val_delta_h_total_t[indices]
            )
            target_delta_h_total_scaled = (
                target_delta_h_total / config.total_enthalpy_delta_scale
            )
            predicted_delta_h_total_scaled = output[
                "delta_h_total_scaled"
            ].reshape(-1).to(torch.float64)
            delta_h_total_loss = F.l1_loss(
                predicted_delta_h_total_scaled,
                target_delta_h_total_scaled,
            )
            delta_h_total_raw_mae = F.l1_loss(
                output["delta_h_total"].reshape(-1),
                target_delta_h_total,
            )
            delta_h_total_transformed_loss = pred_y.sum() * 0.0
            delta_h_total_residual_transformed_loss = pred_y.sum() * 0.0
            if (
                config.total_enthalpy_target_transform
                == TOTAL_ENTHALPY_TARGET_SIGNED_POWER
            ):
                delta_h_total_transformed_loss = F.l1_loss(
                    output["delta_h_total_head_space"]
                    .reshape(-1)
                    .to(torch.float64),
                    _signed_power(
                        target_delta_h_total_scaled,
                        config.total_enthalpy_transform_alpha,
                    ),
                )
            if (
                config.total_enthalpy_residual_mode
                == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
            ):
                baseline_scaled = output[
                    "delta_h_total_baseline_scaled"
                ].reshape(-1).detach()
                residual_target_scaled = (
                    target_delta_h_total_scaled - baseline_scaled
                )
                delta_h_total_residual_transformed_loss = F.l1_loss(
                    output["delta_h_total_residual_head_space"]
                    .reshape(-1)
                    .to(torch.float64),
                    _signed_power(
                        residual_target_scaled,
                        config.total_enthalpy_residual_alpha,
                    ),
                )
        y_loss = F.l1_loss(
            _signed_power(pred_y, config.transform_alpha),
            _signed_power(true_y, config.transform_alpha),
        )
        delta_loss = F.l1_loss(
            _signed_power(pred_y - y0, config.transform_alpha),
            _signed_power(true_y - y0, config.transform_alpha),
        )
        pred_delta = pred_y - y0
        true_delta = true_y - y0
        relative_delta_loss = torch.sum(torch.abs(pred_delta - true_delta)) / torch.sum(
            torch.abs(true_delta)
        ).clamp_min(torch.finfo(true_delta.dtype).tiny)
        pred_transformed = _signed_power(pred_delta, config.transform_alpha)
        true_transformed = _signed_power(true_delta, config.transform_alpha)
        true_norm = torch.linalg.vector_norm(true_transformed, dim=1)
        reactive = true_norm > 1.0e-12
        if torch.any(reactive):
            cosine = F.cosine_similarity(
                pred_transformed[reactive], true_transformed[reactive], dim=1, eps=1.0e-30
            )
            direction_loss = 1.0 - cosine.mean()
        else:
            direction_loss = pred_delta.sum() * 0.0
        if "forward_extent" in output:
            opposing_extent_loss = torch.mean(
                torch.minimum(output["forward_extent"], output["reverse_extent"])
            ) / config.extent_scale
        else:
            opposing_extent_loss = pred_delta.sum() * 0.0
        base_total = (
            config.tp_loss_weight * tp_loss
            + config.state_loss_weight * y_loss
            + config.delta_loss_weight * delta_loss
            + config.relative_delta_loss_weight * relative_delta_loss
            + config.direction_loss_weight * direction_loss
            + config.opposing_extent_loss_weight * opposing_extent_loss
            + config.total_enthalpy_delta_loss_weight * delta_h_total_loss
            + config.total_enthalpy_transformed_loss_weight
            * delta_h_total_transformed_loss
            + config.total_enthalpy_residual_transformed_loss_weight
            * delta_h_total_residual_transformed_loss
        )
        if config.thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
            predicted_temperature = _deployment_endpoint_temperature(
                output,
                prediction,
                current_tp[:, 0],
                state_mean_t,
                state_std_t,
            )
        else:
            # Source and W_mix losses do not depend on this placeholder. A
            # density loss is rejected above because only Fluent may perform
            # the reference-enthalpy temperature inversion in production.
            predicted_temperature = truth[:, 0]
        deployment = _deployment_loss_for_batch(
            loss_config=deployment_loss_config,
            current_y=y0,
            predicted_delta_y=pred_delta,
            target_delta_y=true_delta,
            current_temperature=current_tp[:, 0],
            predicted_temperature=predicted_temperature,
            target_temperature=truth[:, 0],
            pressure=current_tp[:, 1],
            dt=dt_physical,
            molecular_weights=molecular_weights_t,
            zero_species_enthalpy=zero_species_enthalpy,
        )
        total = base_total + deployment["total"]
        thermochemical = {
            "delta_h_total_scaled_loss": delta_h_total_loss,
            "delta_h_total_raw_mae": delta_h_total_raw_mae,
            "delta_h_total_residual_transformed_loss": (
                delta_h_total_residual_transformed_loss
            ),
        }
        return total, output, base_total, deployment, thermochemical

    history = []
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator, device=torch_device)
        total = 0.0
        base_total = 0.0
        deployment_total = 0.0
        train_delta_h_total_scaled_total = 0.0
        train_delta_h_total_raw_total = 0.0
        train_deployment_terms = _empty_loss_accumulator()
        for start in range(0, order.numel(), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss, _output, base_loss, deployment, thermochemical = loss_for(
                indices, True
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * indices.numel()
            base_total += float(base_loss.detach()) * indices.numel()
            deployment_total += float(deployment["total"].detach()) * indices.numel()
            train_delta_h_total_scaled_total += float(
                thermochemical["delta_h_total_scaled_loss"].detach()
            ) * indices.numel()
            train_delta_h_total_raw_total += float(
                thermochemical["delta_h_total_raw_mae"].detach()
            ) * indices.numel()
            _accumulate_loss_report(train_deployment_terms, deployment, indices.numel())
        if jvp_training_data is not None and config.source_jvp_loss_weight > 0.0:
            for pair_ids in deterministic_paired_batch_indices(
                jvp_training_data.train_indices,
                batch_size=config.batch_size,
                seed=config.seed,
                epoch=epoch,
                rank=jvp_rank,
                world_size=jvp_world_size,
                shuffle=True,
            ):
                optimizer.zero_grad(set_to_none=True)
                jvp_report = _jvp_loss_for_indices(
                    model,
                    jvp_training_data,
                    pair_ids,
                    config=config,
                    loss_config=jvp_loss_config,
                    state_mean=state_mean_t,
                    state_std=state_std_t,
                    log_dt_mean=log_dt_mean_t,
                    log_dt_std=log_dt_std_t,
                    device=torch_device,
                    base_affinity=jvp_base_affinity,
                    perturbed_affinity=jvp_perturbed_affinity,
                )
                jvp_report["total"].backward()
                optimizer.step()
        train_loss = total / order.numel()
        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                val_total = 0.0
                val_base_total = 0.0
                val_deployment_total = 0.0
                val_delta_h_total_scaled_total = 0.0
                val_delta_h_total_raw_total = 0.0
                val_deployment_terms = _empty_loss_accumulator()
                negative = 0
                count = 0
                free_energy_positive = 0
                for start in range(0, val_x.shape[0], config.batch_size):
                    indices = torch.arange(
                        start,
                        min(start + config.batch_size, val_x.shape[0]),
                        device=torch_device,
                    )
                    (
                        val_loss,
                        output,
                        val_base_loss,
                        deployment,
                        thermochemical,
                    ) = loss_for(indices, False)
                    val_total += float(val_loss) * indices.numel()
                    val_base_total += float(val_base_loss) * indices.numel()
                    val_deployment_total += float(deployment["total"]) * indices.numel()
                    val_delta_h_total_scaled_total += float(
                        thermochemical["delta_h_total_scaled_loss"]
                    ) * indices.numel()
                    val_delta_h_total_raw_total += float(
                        thermochemical["delta_h_total_raw_mae"]
                    ) * indices.numel()
                    _accumulate_loss_report(val_deployment_terms, deployment, indices.numel())
                    negative += int(torch.sum(output["next_species"] < 0.0))
                    count += output["next_species"].numel()
                    if "local_free_energy_delta" in output:
                        free_energy_positive += int(torch.sum(output["local_free_energy_delta"] > 1e-12))
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_base_loss": base_total / order.numel(),
                    "train_deployment_loss": deployment_total / order.numel(),
                    "validation_loss": val_total / val_x.shape[0],
                    "validation_base_loss": val_base_total / val_x.shape[0],
                    "validation_deployment_loss": val_deployment_total / val_x.shape[0],
                    "negative_entry_rate": negative / count,
                    "positive_free_energy_count": free_energy_positive,
                }
                if jvp_training_data is not None:
                    train_jvp_metrics = _evaluate_jvp_partition(
                        model,
                        jvp_training_data,
                        jvp_training_data.train_indices,
                        prefix="train",
                        config=config,
                        loss_config=jvp_loss_config,
                        state_mean=state_mean_t,
                        state_std=state_std_t,
                        log_dt_mean=log_dt_mean_t,
                        log_dt_std=log_dt_std_t,
                        device=torch_device,
                        rank=jvp_rank,
                        world_size=jvp_world_size,
                        base_affinity=jvp_base_affinity,
                        perturbed_affinity=jvp_perturbed_affinity,
                    )
                    validation_jvp_metrics = _evaluate_jvp_partition(
                        model,
                        jvp_training_data,
                        jvp_training_data.validation_indices,
                        prefix="validation",
                        config=config,
                        loss_config=jvp_loss_config,
                        state_mean=state_mean_t,
                        state_std=state_std_t,
                        log_dt_mean=log_dt_mean_t,
                        log_dt_std=log_dt_std_t,
                        device=torch_device,
                        rank=jvp_rank,
                        world_size=jvp_world_size,
                        base_affinity=jvp_base_affinity,
                        perturbed_affinity=jvp_perturbed_affinity,
                    )
                    row.update(train_jvp_metrics)
                    row.update(validation_jvp_metrics)
                    row["train_loss_with_source_jvp"] = (
                        train_loss
                        + train_jvp_metrics["train_source_jvp_weighted_loss"]
                    )
                    row["validation_loss_with_source_jvp"] = (
                        row["validation_loss"]
                        + validation_jvp_metrics[
                            "validation_source_jvp_weighted_loss"
                        ]
                    )
                if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE:
                    row.update(
                        {
                            "train_delta_h_total_scaled_loss": (
                                train_delta_h_total_scaled_total / order.numel()
                            ),
                            "train_delta_h_total_mae_j_per_kg": (
                                train_delta_h_total_raw_total / order.numel()
                            ),
                            "validation_delta_h_total_scaled_loss": (
                                val_delta_h_total_scaled_total / val_x.shape[0]
                            ),
                            "validation_delta_h_total_mae_j_per_kg": (
                                val_delta_h_total_raw_total / val_x.shape[0]
                            ),
                            "total_enthalpy_delta_scale_j_per_kg": (
                                config.total_enthalpy_delta_scale
                            ),
                        }
                    )
                row.update(
                    _flatten_loss_report(
                        "train",
                        train_deployment_terms,
                        order.numel(),
                        deployment_loss_config,
                    )
                )
                row.update(
                    _flatten_loss_report(
                        "validation",
                        val_deployment_terms,
                        val_x.shape[0],
                        deployment_loss_config,
                    )
                )
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
        "availability_mode": config.availability_mode,
        "availability_p_norm": config.availability_p_norm,
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
        "thermochemical_delta_y_feature_contract": {
            "mode": config.thermochemical_delta_y_feature_mode,
            "equation": (
                "asinh(delta_Y / species_scale)"
                if config.thermochemical_delta_y_feature_mode
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else "sign(delta_Y) * abs(delta_Y) ** 0.1"
            ),
            "scale_source": (
                "training-split-only q90(abs(target_species-current_species))"
                if config.thermochemical_delta_y_feature_mode
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else None
            ),
            "quantile": (
                0.9
                if config.thermochemical_delta_y_feature_mode
                == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
                else None
            ),
            "floor": config.thermochemical_delta_y_scale_floor,
            "species_scales": config.thermochemical_delta_y_scales,
            "consumers": ["auxiliary-temperature", "total-enthalpy"],
            "physical_delta_y_transformed": False,
            "species_reaction_encoder_uses_feature": False,
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
                "target_equation": "delta_h_total = h_total_after - h_total_before",
                "target_units": "J/kg",
                "normalized_target": (
                    "delta_h_total / total_enthalpy_delta_scale"
                ),
                "total_enthalpy_delta_scale_j_per_kg": (
                    config.total_enthalpy_delta_scale
                ),
                "loss": "MAE in fixed-scale normalized delta_h_total space",
                "loss_weight": config.total_enthalpy_delta_loss_weight,
                "predicts_temperature": False,
                "temperature_recovery": (
                    "T_next = Temperature(h_total_before + delta_h_total - "
                    "Reference_Enthalpy(Y_next), Y_next)"
                ),
            }
            if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
            else {
                "mode": DELTA_TEMPERATURE_MODE,
                "predicts_temperature": True,
                "target": "legacy independent delta_temperature",
            }
        ),
        "deployment_loss_contract": {
            "train_label_backend": train_label_backend,
            "validation_label_backend": validation_label_backend,
            "required_backend_when_weighted": FLUENT_NATIVE_DI_BACKEND,
            "temperature_source": (
                "fluent-total-enthalpy-reference-enthalpy-inversion"
                if config.thermochemical_output_mode == DELTA_H_TOTAL_MODE
                else "model-predicted-endpoint"
            ),
            "thermochemical_closure": config.thermochemical_output_mode,
            "terms": {
                name: {
                    "weight": float(getattr(deployment_loss_config, name).weight),
                    "scale": float(getattr(deployment_loss_config, name).scale),
                }
                for name in _DEPLOYMENT_TERM_NAMES
            },
        },
        "jvp_training_contract": (
            None
            if jvp_training_data is None
            else {
                "schema_version": "fluent-di-jvp-v1",
                "label_backend": FLUENT_NATIVE_DI_BACKEND,
                "dataset": config.jvp_dataset,
                "source_map": "g=(Phi_Y(x)-Y)/dt",
                "directional_difference": (
                    "Dg=(g(x+epsilon*v)-g(x))/epsilon; includes -I/dt"
                ),
                "source_jvp_loss_weight": config.source_jvp_loss_weight,
                "source_jvp_loss_scale": config.source_jvp_loss_scale,
                "split_seed": config.seed,
                "validation_fraction": _JVP_VALIDATION_FRACTION,
                "train_pairs": int(jvp_training_data.train_indices.size),
                "validation_pairs": int(
                    jvp_training_data.validation_indices.size
                ),
                "pair_batching": (
                    "deterministic target-independent shuffle; DDP truncation "
                    "without padding or resampling"
                ),
            }
        ),
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
