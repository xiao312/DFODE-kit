from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dfode_kit.physics.smooth_positivity import (
    hard_patankar_process_availability,
    smooth_safe_patankar_process_availability,
)


DELTA_TEMPERATURE_MODE = "delta-temperature"
DELTA_H_TOTAL_MODE = "delta-h-total"
THERMOCHEMICAL_OUTPUT_MODES = (
    DELTA_TEMPERATURE_MODE,
    DELTA_H_TOTAL_MODE,
)
TOTAL_ENTHALPY_TARGET_IDENTITY = "identity"
TOTAL_ENTHALPY_TARGET_SIGNED_POWER = "signed-power"
TOTAL_ENTHALPY_TARGET_TRANSFORMS = (
    TOTAL_ENTHALPY_TARGET_IDENTITY,
    TOTAL_ENTHALPY_TARGET_SIGNED_POWER,
)
TOTAL_ENTHALPY_RESIDUAL_NONE = "none"
TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER = "signed-power"
TOTAL_ENTHALPY_RESIDUAL_MODES = (
    TOTAL_ENTHALPY_RESIDUAL_NONE,
    TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER,
)
THERMOCHEMICAL_HEAD_INPUT_LEGACY = "latent-delta-y"
THERMOCHEMICAL_HEAD_INPUT_STATE_TIME = "latent-state-time-delta-y"
THERMOCHEMICAL_HEAD_INPUT_MODES = (
    THERMOCHEMICAL_HEAD_INPUT_LEGACY,
    THERMOCHEMICAL_HEAD_INPUT_STATE_TIME,
)
DEFAULT_THERMOCHEMICAL_STATE_LOW_SCALE = float(2**24)
THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY = "legacy-signed-power"
THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH = "asinh-training-q90"
THERMOCHEMICAL_DELTA_Y_FEATURE_MODES = (
    THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY,
    THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH,
)
THERMOCHEMICAL_PROCESS_FEATURE_NONE = "none"
THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER = (
    "species-gross-turnover"
)
THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS = "process-extents"
THERMOCHEMICAL_PROCESS_FEATURE_MODES = (
    THERMOCHEMICAL_PROCESS_FEATURE_NONE,
    THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER,
    THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS,
)


def inverse_signed_power(values: torch.Tensor, alpha: float) -> torch.Tensor:
    """Invert sign(x) * abs(x)**alpha without an epsilon offset."""
    return torch.sign(values) * torch.pow(torch.abs(values), 1.0 / alpha)


def _backbone(input_dim: int, hidden_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim + 1, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, latent_dim),
        nn.SiLU(),
    )


class NeuralPatankarIntervalModel(nn.Module):
    """Neural reaction-process proposal with a positive resource allocation layer.

    Forward and reversible backward reaction demands are nonnegative. Competing
    processes share each reactant's available mass through a common depletion
    ratio. This is a first-order process-based neural Patankar construction:
    positivity and stoichiometric conservation hold for every network output.
    """

    def __init__(
        self,
        input_dim: int,
        reactant_mass_matrix: np.ndarray,
        product_mass_matrix: np.ndarray,
        reaction_reversible: np.ndarray,
        *,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        extent_scale: float = 1e-4,
        demand_bias_init: float = -8.0,
        availability_floor: float = 1e-30,
        temperature_delta_scale: float = 10.0,
        thermochemical_output_mode: str = DELTA_TEMPERATURE_MODE,
        total_enthalpy_delta_scale: float = 1.0e3,
        total_enthalpy_target_transform: str = (
            TOTAL_ENTHALPY_TARGET_IDENTITY
        ),
        total_enthalpy_transform_alpha: float = 0.1,
        total_enthalpy_residual_mode: str = TOTAL_ENTHALPY_RESIDUAL_NONE,
        total_enthalpy_residual_alpha: float = 0.1,
        thermochemical_head_input_mode: str = THERMOCHEMICAL_HEAD_INPUT_LEGACY,
        thermochemical_head_hidden_dim: int | None = None,
        thermochemical_head_depth: int = 1,
        total_enthalpy_head_hidden_dim: int = 0,
        total_enthalpy_head_depth: int = 0,
        thermochemical_aux_temperature: bool = False,
        thermochemical_delta_y_feature_mode: str = (
            THERMOCHEMICAL_DELTA_Y_FEATURE_LEGACY
        ),
        thermochemical_delta_y_scales: np.ndarray | None = None,
        thermochemical_process_feature_mode: str = (
            THERMOCHEMICAL_PROCESS_FEATURE_NONE
        ),
        thermochemical_state_high_low: bool = False,
        thermochemical_state_low_scale: float = (
            DEFAULT_THERMOCHEMICAL_STATE_LOW_SCALE
        ),
        availability_mode: str = "hard",
        availability_p_norm: float = 32.0,
    ):
        super().__init__()
        reactants = np.asarray(reactant_mass_matrix, dtype=np.float64)
        products = np.asarray(product_mass_matrix, dtype=np.float64)
        reversible = np.asarray(reaction_reversible, dtype=np.float64)
        reverse_reactants = products * reversible[None, :]
        reverse_products = reactants * reversible[None, :]
        consumption = np.concatenate([reactants, reverse_reactants], axis=1)
        production = np.concatenate([products, reverse_products], axis=1)
        self.register_buffer("consumption", torch.tensor(consumption, dtype=torch.float64))
        self.register_buffer("process_stoich", torch.tensor(production - consumption, dtype=torch.float64))
        self.register_buffer(
            "process_turnover_stoich",
            torch.tensor(
                2.0 * consumption + (production - consumption),
                dtype=torch.float64,
            ),
            persistent=False,
        )
        self.register_buffer("reaction_reversible", torch.tensor(reversible, dtype=torch.float64))
        self.register_buffer(
            "process_gate",
            torch.cat(
                [
                    torch.ones(reactants.shape[1], dtype=torch.float64),
                    torch.tensor(reversible, dtype=torch.float64),
                ]
            ),
            persistent=False,
        )
        self.extent_scale = float(extent_scale)
        self.availability_floor = float(availability_floor)
        if thermochemical_output_mode not in THERMOCHEMICAL_OUTPUT_MODES:
            raise ValueError(
                "thermochemical_output_mode must be one of "
                f"{THERMOCHEMICAL_OUTPUT_MODES}"
            )
        if not np.isfinite(total_enthalpy_delta_scale) or total_enthalpy_delta_scale <= 0.0:
            raise ValueError("total_enthalpy_delta_scale must be finite and positive")
        self.thermochemical_output_mode = thermochemical_output_mode
        self.total_enthalpy_delta_scale = float(total_enthalpy_delta_scale)
        if total_enthalpy_target_transform not in (
            TOTAL_ENTHALPY_TARGET_TRANSFORMS
        ):
            raise ValueError(
                "total_enthalpy_target_transform must be one of "
                f"{TOTAL_ENTHALPY_TARGET_TRANSFORMS}"
            )
        if (
            total_enthalpy_target_transform
            != TOTAL_ENTHALPY_TARGET_IDENTITY
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "transformed total-enthalpy targets require delta-h-total mode"
            )
        if (
            not np.isfinite(total_enthalpy_transform_alpha)
            or not 0.0 < total_enthalpy_transform_alpha <= 1.0
        ):
            raise ValueError(
                "total_enthalpy_transform_alpha must be finite and in (0, 1]"
            )
        self.total_enthalpy_target_transform = str(
            total_enthalpy_target_transform
        )
        self.total_enthalpy_transform_alpha = float(
            total_enthalpy_transform_alpha
        )
        if total_enthalpy_residual_mode not in TOTAL_ENTHALPY_RESIDUAL_MODES:
            raise ValueError(
                "total_enthalpy_residual_mode must be one of "
                f"{TOTAL_ENTHALPY_RESIDUAL_MODES}"
            )
        if (
            total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "total-enthalpy residual correction requires delta-h-total mode"
            )
        if (
            total_enthalpy_residual_mode != TOTAL_ENTHALPY_RESIDUAL_NONE
            and total_enthalpy_target_transform
            != TOTAL_ENTHALPY_TARGET_IDENTITY
        ):
            raise ValueError(
                "total-enthalpy residual correction requires an identity "
                "physical baseline head"
            )
        if (
            not np.isfinite(total_enthalpy_residual_alpha)
            or not 0.0 < total_enthalpy_residual_alpha <= 1.0
        ):
            raise ValueError(
                "total_enthalpy_residual_alpha must be finite and in (0, 1]"
            )
        self.total_enthalpy_residual_mode = str(
            total_enthalpy_residual_mode
        )
        self.total_enthalpy_residual_alpha = float(
            total_enthalpy_residual_alpha
        )
        if thermochemical_head_input_mode not in THERMOCHEMICAL_HEAD_INPUT_MODES:
            raise ValueError(
                "thermochemical_head_input_mode must be one of "
                f"{THERMOCHEMICAL_HEAD_INPUT_MODES}"
            )
        self.thermochemical_head_input_mode = thermochemical_head_input_mode
        self.thermochemical_aux_temperature = bool(
            thermochemical_aux_temperature
        )
        if thermochemical_delta_y_feature_mode not in (
            THERMOCHEMICAL_DELTA_Y_FEATURE_MODES
        ):
            raise ValueError(
                "thermochemical_delta_y_feature_mode must be one of "
                f"{THERMOCHEMICAL_DELTA_Y_FEATURE_MODES}"
            )
        self.thermochemical_delta_y_feature_mode = str(
            thermochemical_delta_y_feature_mode
        )
        if thermochemical_delta_y_scales is None:
            delta_y_scales = np.ones(reactants.shape[0], dtype=np.float64)
        else:
            delta_y_scales = np.asarray(
                thermochemical_delta_y_scales, dtype=np.float64
            )
        if delta_y_scales.shape != (reactants.shape[0],):
            raise ValueError(
                "thermochemical_delta_y_scales must contain one value per species"
            )
        if not np.all(np.isfinite(delta_y_scales)) or np.any(
            delta_y_scales <= 0.0
        ):
            raise ValueError(
                "thermochemical_delta_y_scales must be finite and positive"
            )
        if (
            self.thermochemical_delta_y_feature_mode
            == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
            and thermochemical_delta_y_scales is None
        ):
            raise ValueError(
                "asinh thermochemical delta-Y features require fixed scales"
            )
        if (
            self.thermochemical_delta_y_feature_mode
            == THERMOCHEMICAL_DELTA_Y_FEATURE_ASINH
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "asinh thermochemical delta-Y features require "
                "delta-h-total output mode"
            )
        self.register_buffer(
            "thermochemical_delta_y_scales",
            torch.as_tensor(delta_y_scales, dtype=torch.float64),
            persistent=False,
        )
        if thermochemical_process_feature_mode not in (
            THERMOCHEMICAL_PROCESS_FEATURE_MODES
        ):
            raise ValueError(
                "thermochemical_process_feature_mode must be one of "
                f"{THERMOCHEMICAL_PROCESS_FEATURE_MODES}"
            )
        if (
            thermochemical_process_feature_mode
            != THERMOCHEMICAL_PROCESS_FEATURE_NONE
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "thermochemical process features require delta-h-total "
                "output mode"
            )
        self.thermochemical_process_feature_mode = str(
            thermochemical_process_feature_mode
        )
        self.thermochemical_state_high_low = bool(
            thermochemical_state_high_low
        )
        if (
            not np.isfinite(thermochemical_state_low_scale)
            or thermochemical_state_low_scale <= 0.0
        ):
            raise ValueError(
                "thermochemical_state_low_scale must be finite and positive"
            )
        self.thermochemical_state_low_scale = float(
            thermochemical_state_low_scale
        )
        if (
            self.thermochemical_aux_temperature
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "thermochemical auxiliary temperature requires "
                "delta-h-total output mode"
            )
        if (
            self.thermochemical_state_high_low
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "thermochemical high/low state features require "
                "delta-h-total output mode"
            )
        if (
            self.thermochemical_state_high_low
            and thermochemical_head_input_mode
            != THERMOCHEMICAL_HEAD_INPUT_STATE_TIME
        ):
            raise ValueError(
                "thermochemical high/low state features require the "
                "state-time thermochemical head input mode"
            )
        if availability_mode not in {"hard", "smooth-safe"}:
            raise ValueError("availability_mode must be 'hard' or 'smooth-safe'")
        self.availability_mode = availability_mode
        self.availability_p_norm = float(availability_p_norm)
        self.n_reactions = reactants.shape[1]
        self.encoder = _backbone(input_dim, hidden_dim, latent_dim)
        thermochemical_input_dim = latent_dim + reactants.shape[0]
        if thermochemical_head_input_mode == THERMOCHEMICAL_HEAD_INPUT_STATE_TIME:
            thermochemical_input_dim += input_dim + 1
        head_hidden_dim = int(thermochemical_head_hidden_dim or hidden_dim)
        if head_hidden_dim <= 0:
            raise ValueError("thermochemical_head_hidden_dim must be positive")
        if thermochemical_head_depth < 1:
            raise ValueError("thermochemical_head_depth must be at least one")
        if total_enthalpy_head_hidden_dim < 0:
            raise ValueError(
                "total_enthalpy_head_hidden_dim must be nonnegative"
            )
        if total_enthalpy_head_depth < 0:
            raise ValueError("total_enthalpy_head_depth must be nonnegative")
        if (
            (total_enthalpy_head_hidden_dim or total_enthalpy_head_depth)
            and thermochemical_output_mode != DELTA_H_TOTAL_MODE
        ):
            raise ValueError(
                "total-enthalpy head overrides require delta-h-total mode"
            )
        resolved_total_enthalpy_head_hidden_dim = int(
            total_enthalpy_head_hidden_dim or head_hidden_dim
        )
        resolved_total_enthalpy_head_depth = int(
            total_enthalpy_head_depth or thermochemical_head_depth
        )
        self.total_enthalpy_head_hidden_dim = (
            resolved_total_enthalpy_head_hidden_dim
        )
        self.total_enthalpy_head_depth = resolved_total_enthalpy_head_depth

        def build_thermochemical_head(
            head_input_dim: int,
            *,
            build_hidden_dim: int,
            build_depth: int,
        ) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Linear(head_input_dim, build_hidden_dim),
                nn.SiLU(),
            ]
            for _ in range(build_depth - 1):
                layers.extend(
                    [
                        nn.Linear(build_hidden_dim, build_hidden_dim),
                        nn.SiLU(),
                    ]
                )
            layers.append(nn.Linear(build_hidden_dim, 1))
            return nn.Sequential(*layers)

        self.temperature_delta_scale = float(temperature_delta_scale)
        if self.thermochemical_aux_temperature:
            auxiliary_input_dim = thermochemical_input_dim
            if self.thermochemical_state_high_low:
                auxiliary_input_dim += input_dim
            self.thermochemical_aux_temperature_head = (
                build_thermochemical_head(
                    auxiliary_input_dim,
                    build_hidden_dim=head_hidden_dim,
                    build_depth=thermochemical_head_depth,
                )
            )
            thermochemical_input_dim += 1
        if self.thermochemical_state_high_low:
            thermochemical_input_dim += input_dim
        if (
            self.thermochemical_process_feature_mode
            == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
        ):
            thermochemical_input_dim += reactants.shape[0]
        elif (
            self.thermochemical_process_feature_mode
            == THERMOCHEMICAL_PROCESS_FEATURE_PROCESS_EXTENTS
        ):
            thermochemical_input_dim += 2 * self.n_reactions
        if thermochemical_output_mode == DELTA_H_TOTAL_MODE:
            thermochemical_head = build_thermochemical_head(
                thermochemical_input_dim,
                build_hidden_dim=resolved_total_enthalpy_head_hidden_dim,
                build_depth=resolved_total_enthalpy_head_depth,
            )
        else:
            thermochemical_head = build_thermochemical_head(
                thermochemical_input_dim,
                build_hidden_dim=head_hidden_dim,
                build_depth=thermochemical_head_depth,
            )
        if thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
            # Keep legacy names and module structure so existing checkpoints load
            # without migration.
            self.tp_head = nn.Linear(latent_dim, 2)
            self.temperature_delta_head = thermochemical_head
        else:
            # No temperature or TP prediction head exists in this mode. Fluent
            # owns temperature recovery from total enthalpy and composition.
            self.total_enthalpy_delta_head = thermochemical_head
            if (
                self.total_enthalpy_residual_mode
                == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
            ):
                self.total_enthalpy_residual_head = build_thermochemical_head(
                    thermochemical_input_dim,
                    build_hidden_dim=resolved_total_enthalpy_head_hidden_dim,
                    build_depth=resolved_total_enthalpy_head_depth,
                )
                nn.init.zeros_(self.total_enthalpy_residual_head[-1].weight)
                nn.init.zeros_(self.total_enthalpy_residual_head[-1].bias)
                for parameter in self.total_enthalpy_delta_head.parameters():
                    parameter.requires_grad_(False)
        self.demand_head = nn.Linear(latent_dim, 2 * self.n_reactions)
        nn.init.zeros_(thermochemical_head[-1].weight)
        nn.init.zeros_(thermochemical_head[-1].bias)
        if self.thermochemical_aux_temperature:
            nn.init.zeros_(self.thermochemical_aux_temperature_head[-1].weight)
            nn.init.zeros_(self.thermochemical_aux_temperature_head[-1].bias)
        nn.init.constant_(self.demand_head.bias, float(demand_bias_init))

    def forward(
        self,
        current_state,
        log_dt,
        *,
        current_species,
        thermochemical_state_low=None,
    ):
        if self.thermochemical_state_high_low:
            if thermochemical_state_low is None:
                raise ValueError(
                    "thermochemical_state_low is required when high/low "
                    "state features are enabled"
                )
            if thermochemical_state_low.shape != current_state.shape:
                raise ValueError(
                    "thermochemical_state_low must match current_state shape"
                )
        hidden = self.encoder(torch.cat([current_state, log_dt], dim=-1))
        demand = (
            F.softplus(self.demand_head(hidden).to(torch.float64))
            * self.extent_scale
            * self.process_gate
        )
        if self.availability_mode == "smooth-safe":
            allocation = smooth_safe_patankar_process_availability(
                current_species,
                demand,
                self.consumption,
                availability_floor=self.availability_floor,
                p_norm=self.availability_p_norm,
            )
        else:
            allocation = hard_patankar_process_availability(
                current_species,
                demand,
                self.consumption,
                availability_floor=self.availability_floor,
            )
        process_availability = allocation.process_scale
        process_availability = process_availability * (1.0 - 1e-12)
        process_extent = demand * process_availability
        delta_y = process_extent @ self.process_stoich.T
        thermochemical_process_features = None
        if (
            self.thermochemical_process_feature_mode
            == THERMOCHEMICAL_PROCESS_FEATURE_SPECIES_GROSS_TURNOVER
        ):
            species_gross_turnover = (
                process_extent @ self.process_turnover_stoich.T
            )
            thermochemical_process_features = torch.pow(
                species_gross_turnover.clamp_min(0.0), 0.1
            ).to(hidden.dtype)
        elif (
            self.thermochemical_process_feature_mode
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
        next_y = current_species.to(torch.float64) + delta_y
        forward_extent = process_extent[:, : self.n_reactions]
        reverse_extent = process_extent[:, self.n_reactions :]
        result = {
            "next_species": next_y,
            "delta_species": delta_y,
            "reaction_extent": forward_extent - reverse_extent,
            "forward_extent": forward_extent,
            "reverse_extent": reverse_extent,
            "process_availability": process_availability,
            "process_active_species": allocation.active_species,
            "minimum_process_availability": process_availability.amin(dim=1, keepdim=True),
        }
        thermochemical_feature_blocks = [hidden]
        if (
            self.thermochemical_head_input_mode
            == THERMOCHEMICAL_HEAD_INPUT_STATE_TIME
        ):
            thermochemical_feature_blocks.extend([current_state, log_dt])
        thermochemical_feature_blocks.append(temperature_species_features)
        thermochemical_features = torch.cat(thermochemical_feature_blocks, dim=1)
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
            result["aux_delta_temperature_scaled"] = (
                auxiliary_delta_temperature_scaled
            )
            result["aux_delta_temperature"] = (
                auxiliary_delta_temperature_scaled.to(torch.float64)
                * self.temperature_delta_scale
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
        if self.thermochemical_output_mode == DELTA_TEMPERATURE_MODE:
            next_tp = self.tp_head(hidden).to(torch.float64)
            delta_temperature = (
                self.temperature_delta_head(thermochemical_features).to(torch.float64)
                * self.temperature_delta_scale
            )
            result["next_state"] = torch.cat([next_tp, next_y], dim=-1)
            result["delta_temperature"] = delta_temperature
        else:
            delta_h_total_head_space = self.total_enthalpy_delta_head(
                thermochemical_features
            )
            if (
                self.total_enthalpy_residual_mode
                == TOTAL_ENTHALPY_RESIDUAL_SIGNED_POWER
            ):
                delta_h_total_baseline_scaled = (
                    delta_h_total_head_space.to(torch.float64)
                )
                delta_h_total_residual_head_space = (
                    self.total_enthalpy_residual_head(
                        thermochemical_features
                    )
                )
                delta_h_total_correction_scaled = inverse_signed_power(
                    delta_h_total_residual_head_space.to(torch.float64),
                    self.total_enthalpy_residual_alpha,
                )
                delta_h_total_scaled = (
                    delta_h_total_baseline_scaled
                    + delta_h_total_correction_scaled
                )
                result["delta_h_total_baseline_scaled"] = (
                    delta_h_total_baseline_scaled
                )
                result["delta_h_total_baseline"] = (
                    delta_h_total_baseline_scaled
                    * self.total_enthalpy_delta_scale
                )
                result["delta_h_total_residual_head_space"] = (
                    delta_h_total_residual_head_space
                )
                result["delta_h_total_correction_scaled"] = (
                    delta_h_total_correction_scaled
                )
                result["delta_h_total_correction"] = (
                    delta_h_total_correction_scaled
                    * self.total_enthalpy_delta_scale
                )
            elif (
                self.total_enthalpy_target_transform
                == TOTAL_ENTHALPY_TARGET_SIGNED_POWER
            ):
                delta_h_total_scaled = inverse_signed_power(
                    delta_h_total_head_space.to(torch.float64),
                    self.total_enthalpy_transform_alpha,
                )
            else:
                delta_h_total_scaled = delta_h_total_head_space.to(
                    torch.float64
                )
            result["next_state"] = torch.cat(
                [current_state[:, :2].to(torch.float64), next_y], dim=-1
            )
            result["delta_h_total_head_space"] = delta_h_total_head_space
            result["delta_h_total_scaled"] = delta_h_total_scaled
            result["delta_h_total"] = (
                delta_h_total_scaled * self.total_enthalpy_delta_scale
            )
        return result


class ReactionTrajectoryFreeEnergyModel(nn.Module):
    """Reduced reaction-trajectory proximal model.

    The network predicts an initial reduced reaction coordinate, mobility, and
    diagonal preconditioner. Fixed proximal iterations remain in the range of
    the molar stoichiometric matrix, use fraction-to-boundary positivity, and
    backtrack until an affinity-consistent local ideal-mixture Gibbs functional
    does not increase.
    """

    def __init__(
        self,
        input_dim: int,
        stoichiometric_matrix: np.ndarray,
        molecular_weights: np.ndarray,
        *,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        trajectory_scale: float = 1e-5,
        mobility_scale: float = 1e-4,
        proximal_steps: int = 3,
        proximal_beta: float = 1.0,
        positivity_floor: float = 1e-30,
        positivity_safety: float = 0.99,
        free_energy_backtracks: int = 16,
        svd_tolerance: float = 1e-10,
    ):
        super().__init__()
        stoich = np.asarray(stoichiometric_matrix, dtype=np.float64)
        u, singular_values, vh = np.linalg.svd(stoich, full_matrices=False)
        rank = int(np.sum(singular_values > svd_tolerance * singular_values[0]))
        basis = u[:, :rank]
        affinity_to_reduced = -(vh[:rank].T / singular_values[:rank][None, :])
        self.register_buffer("reaction_basis", torch.tensor(basis, dtype=torch.float64))
        self.register_buffer(
            "affinity_to_reduced",
            torch.tensor(affinity_to_reduced, dtype=torch.float64),
        )
        self.register_buffer(
            "molecular_weights",
            torch.tensor(np.asarray(molecular_weights), dtype=torch.float64),
        )
        self.reduced_dim = rank
        self.trajectory_scale = float(trajectory_scale)
        self.mobility_scale = float(mobility_scale)
        self.proximal_steps = int(proximal_steps)
        self.proximal_beta = float(proximal_beta)
        self.positivity_floor = float(positivity_floor)
        self.positivity_safety = float(positivity_safety)
        self.free_energy_backtracks = int(free_energy_backtracks)
        self.encoder = _backbone(input_dim, hidden_dim, latent_dim)
        self.thermo_encoder = nn.Sequential(
            nn.Linear(stoich.shape[1], hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.tp_head = nn.Linear(latent_dim, 2)
        self.trajectory_head = nn.Linear(latent_dim, rank)
        self.mobility_head = nn.Linear(latent_dim, rank)
        self.preconditioner_head = nn.Linear(latent_dim, rank)

    def _free_energy(self, concentration, standard_offset):
        c = concentration.clamp_min(self.positivity_floor)
        return torch.sum(c * (torch.log(c) - 1.0 + standard_offset), dim=-1)

    def forward(self, current_state, log_dt, affinity_hat, *, current_species):
        hidden = self.encoder(torch.cat([current_state, log_dt], dim=-1))
        hidden = hidden + self.thermo_encoder(affinity_hat)
        next_tp = self.tp_head(hidden).to(torch.float64)
        proposal = torch.tanh(self.trajectory_head(hidden).to(torch.float64)) * self.trajectory_scale
        mobility = F.softplus(self.mobility_head(hidden).to(torch.float64)) * self.mobility_scale
        mobility = mobility.clamp_min(1e-16)
        preconditioner = 0.1 + 1.8 * torch.sigmoid(self.preconditioner_head(hidden).to(torch.float64))

        current_c = current_species.to(torch.float64) / self.molecular_weights
        current_c = current_c.clamp_min(self.positivity_floor)
        reduced_gradient = affinity_hat.to(torch.float64) @ self.affinity_to_reduced
        log_current = torch.log(current_c)
        standard_offset = (
            reduced_gradient - log_current @ self.reaction_basis
        ) @ self.reaction_basis.T
        eta = torch.zeros_like(proposal)

        for _ in range(self.proximal_steps):
            concentration = current_c + eta @ self.reaction_basis.T
            log_c = torch.log(concentration.clamp_min(self.positivity_floor))
            free_energy_gradient = (log_c + standard_offset) @ self.reaction_basis
            gradient = (eta - proposal) / mobility + self.proximal_beta * free_energy_gradient
            curvature = (
                1.0 / mobility
                + self.proximal_beta
                * (self.reaction_basis.square()[None] / concentration[:, :, None].clamp_min(self.positivity_floor)).sum(dim=1)
            )
            step = -preconditioner * gradient / curvature.clamp_min(1e-30)
            delta_c = step @ self.reaction_basis.T
            boundary = torch.where(
                delta_c < 0.0,
                self.positivity_safety
                * (concentration - self.positivity_floor).clamp_min(0.0)
                / (-delta_c).clamp_min(1e-300),
                torch.full_like(delta_c, float("inf")),
            )
            alpha = torch.minimum(
                torch.ones((eta.shape[0], 1), dtype=torch.float64, device=eta.device),
                boundary.amin(dim=1, keepdim=True),
            )
            eta = eta + alpha * step

        free_energy_initial = self._free_energy(current_c, standard_offset)
        for _ in range(self.free_energy_backtracks):
            candidate_c = current_c + eta @ self.reaction_basis.T
            free_energy_candidate = self._free_energy(candidate_c, standard_offset)
            invalid = (
                torch.any(candidate_c <= self.positivity_floor, dim=1)
                | (free_energy_candidate > free_energy_initial)
            )
            eta = torch.where(invalid[:, None], 0.5 * eta, eta)

        next_c = current_c + eta @ self.reaction_basis.T
        next_y = next_c * self.molecular_weights
        delta_y = next_y - current_species.to(torch.float64)
        free_energy_delta = self._free_energy(next_c, standard_offset) - free_energy_initial
        return {
            "next_state": torch.cat([next_tp, next_y], dim=-1),
            "next_species": next_y,
            "delta_species": delta_y,
            "reduced_reaction_coordinate": eta,
            "reaction_mobility": mobility,
            "reaction_preconditioner": preconditioner,
            "local_free_energy_delta": free_energy_delta[:, None],
        }
