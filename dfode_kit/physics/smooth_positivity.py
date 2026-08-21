from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReactionExtentScalingResult:
    """Outputs of a scalar reaction-coordinate positivity limiter."""

    proposed_extent: torch.Tensor
    proposed_delta_y: torch.Tensor
    scale: torch.Tensor
    scaled_extent: torch.Tensor
    delta_y: torch.Tensor
    next_y: torch.Tensor
    hard_cap: torch.Tensor
    active_species: torch.Tensor


@dataclass(frozen=True)
class PatankarProcessAvailabilityResult:
    """Resource-allocation outputs for Patankar reaction processes."""

    demand: torch.Tensor
    requested_species: torch.Tensor
    species_availability: torch.Tensor
    process_scale: torch.Tensor
    process_extent: torch.Tensor
    active_species: torch.Tensor


def _float64_matrix(matrix, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        return matrix.to(device=reference.device, dtype=torch.float64)
    return torch.as_tensor(matrix, device=reference.device, dtype=torch.float64)


def reaction_coordinate_delta(
    reaction_extent: torch.Tensor,
    stoichiometric_mass_matrix,
) -> torch.Tensor:
    """Return ``delta_Y = xi @ (W S).T`` for dense or sparse ``W S``."""

    extent = reaction_extent.to(torch.float64)
    matrix = _float64_matrix(stoichiometric_mass_matrix, extent)
    if extent.ndim != 2 or matrix.ndim != 2:
        raise ValueError("reaction_extent and stoichiometric_mass_matrix must be 2D")
    if extent.shape[1] != matrix.shape[1]:
        raise ValueError("reaction extent width does not match reaction count")
    if matrix.layout == torch.strided:
        return extent @ matrix.T
    matrix_coo = matrix if matrix.layout == torch.sparse_coo else matrix.to_sparse_coo()
    return torch.sparse.mm(matrix_coo, extent.T).T


def hard_fraction_to_boundary(
    current_y: torch.Tensor,
    proposed_delta_y: torch.Tensor,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the current hard global limiter and its limiting species."""

    y0 = current_y.to(torch.float64)
    delta = proposed_delta_y.to(device=y0.device, dtype=torch.float64)
    if y0.shape != delta.shape or y0.ndim != 2:
        raise ValueError("current_y and proposed_delta_y must have the same 2D shape")
    if numerical_tolerance < 0.0:
        raise ValueError("numerical_tolerance must be nonnegative")
    margin = (y0 - float(floor) + float(numerical_tolerance)).clamp_min(0.0)
    ratios = torch.where(
        delta < 0.0,
        margin / (-delta).clamp_min(torch.finfo(torch.float64).tiny),
        torch.full_like(delta, torch.inf),
    )
    minimum_ratio, minimum_index = torch.min(ratios, dim=-1, keepdim=True)
    cap = torch.minimum(torch.ones_like(minimum_ratio), minimum_ratio).clamp(0.0, 1.0)
    active = torch.where(
        cap[:, 0] < 1.0,
        minimum_index[:, 0],
        torch.full_like(minimum_index[:, 0], -1),
    )
    return cap, active


def _reaction_result(
    proposed_extent: torch.Tensor,
    proposed_delta_y: torch.Tensor,
    current_y: torch.Tensor,
    stoichiometric_mass_matrix,
    scale: torch.Tensor,
    hard_cap: torch.Tensor,
    active_species: torch.Tensor,
) -> ReactionExtentScalingResult:
    extent64 = proposed_extent.to(torch.float64)
    y0 = current_y.to(device=extent64.device, dtype=torch.float64)
    scaled_extent = extent64 * scale
    delta_y = reaction_coordinate_delta(scaled_extent, stoichiometric_mass_matrix)
    return ReactionExtentScalingResult(
        proposed_extent=extent64,
        proposed_delta_y=proposed_delta_y,
        scale=scale,
        scaled_extent=scaled_extent,
        delta_y=delta_y,
        next_y=y0 + delta_y,
        hard_cap=hard_cap,
        active_species=active_species,
    )


def hard_reaction_extent_scaling(
    current_y: torch.Tensor,
    proposed_extent: torch.Tensor,
    stoichiometric_mass_matrix,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
) -> ReactionExtentScalingResult:
    """Apply the existing hard fraction-to-boundary rule in reaction space."""

    proposed_delta = reaction_coordinate_delta(proposed_extent, stoichiometric_mass_matrix)
    cap, active = hard_fraction_to_boundary(
        current_y,
        proposed_delta,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
    )
    return _reaction_result(
        proposed_extent,
        proposed_delta,
        current_y,
        stoichiometric_mass_matrix,
        cap,
        cap,
        active,
    )


def smooth_safe_reaction_extent_scaling(
    current_y: torch.Tensor,
    proposed_extent: torch.Tensor,
    stoichiometric_mass_matrix,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 1e-15,
    p_norm: float = 32.0,
    transition_width: float = 1e-3,
    roundoff_safety_ulps: float = 8.0,
) -> ReactionExtentScalingResult:
    r"""Apply a differentiable lower bound to the hard global scale.

    Let ``m_i = Y_i - floor + tolerance`` and
    ``q_i = -delta_Y_i / m_i``. The hard admissible scale is

    ``lambda_hard = 1 / max(1, max_i relu(q_i))``.

    ``q_tilde_i = width * softplus(q_i / width)`` is a smooth upper bound on
    ``relu(q_i)``. Therefore

    ``lambda = (1 + sum_i q_tilde_i**p)**(-1/p) <= lambda_hard``.

    Scaling the complete reaction-extent vector by this scalar keeps the
    update in ``Range(W S)`` and guarantees
    ``Y_next >= floor - numerical_tolerance`` for an initially admissible
    state. The p-norm is evaluated in log space for multiscale compositions.
    """

    if p_norm < 1.0:
        raise ValueError("p_norm must be at least one")
    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive")
    if numerical_tolerance < 0.0:
        raise ValueError("numerical_tolerance must be nonnegative")
    if roundoff_safety_ulps < 0.0:
        raise ValueError("roundoff_safety_ulps must be nonnegative")

    proposed_delta = reaction_coordinate_delta(proposed_extent, stoichiometric_mass_matrix)
    y0 = current_y.to(device=proposed_delta.device, dtype=torch.float64)
    if y0.shape != proposed_delta.shape:
        raise ValueError("current_y shape does not match the species update")
    hard_cap, active = hard_fraction_to_boundary(
        y0,
        proposed_delta,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
    )
    margin = y0 - float(floor) + float(numerical_tolerance)
    ratio = -proposed_delta / margin.clamp_min(torch.finfo(torch.float64).tiny)
    smooth_consumption = float(transition_width) * F.softplus(
        ratio / float(transition_width),
        threshold=36.0,
    )
    log_terms = float(p_norm) * torch.log(
        smooth_consumption.clamp_min(torch.finfo(torch.float64).tiny)
    )
    baseline = torch.zeros(
        (log_terms.shape[0], 1),
        dtype=torch.float64,
        device=log_terms.device,
    )
    log_upper_max = torch.logsumexp(
        torch.cat([baseline, log_terms], dim=-1),
        dim=-1,
        keepdim=True,
    ) / float(p_norm)
    scale = torch.exp(-log_upper_max)
    scale = scale * (
        1.0 - float(roundoff_safety_ulps) * torch.finfo(torch.float64).eps
    )
    # In exact arithmetic this branch is inactive. It catches only numerical
    # underflow or an exactly active zero-margin boundary.
    scale = torch.where(scale > hard_cap, hard_cap, scale).clamp(0.0, 1.0)
    return _reaction_result(
        proposed_extent,
        proposed_delta,
        y0,
        stoichiometric_mass_matrix,
        scale,
        hard_cap,
        active,
    )


class SmoothReactionExtentLimiter(torch.nn.Module):
    """Reusable float64 constraint layer for a fixed ``W S`` matrix."""

    def __init__(
        self,
        stoichiometric_mass_matrix,
        *,
        floor: float = 0.0,
        numerical_tolerance: float = 1e-15,
        p_norm: float = 32.0,
        transition_width: float = 1e-3,
    ):
        super().__init__()
        matrix = torch.as_tensor(stoichiometric_mass_matrix, dtype=torch.float64)
        if matrix.ndim != 2:
            raise ValueError("stoichiometric_mass_matrix must be 2D")
        self.register_buffer("stoichiometric_mass_matrix", matrix)
        self.floor = float(floor)
        self.numerical_tolerance = float(numerical_tolerance)
        self.p_norm = float(p_norm)
        self.transition_width = float(transition_width)

    def forward(
        self,
        proposed_extent: torch.Tensor,
        current_y: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        result = smooth_safe_reaction_extent_scaling(
            current_y,
            proposed_extent,
            self.stoichiometric_mass_matrix,
            floor=self.floor,
            numerical_tolerance=self.numerical_tolerance,
            p_norm=self.p_norm,
            transition_width=self.transition_width,
        )
        return {
            "proposed_reaction_extent": result.proposed_extent,
            "proposed_delta_y": result.proposed_delta_y,
            "limiter": result.scale,
            "hard_limiter_cap": result.hard_cap,
            "reaction_extent": result.scaled_extent,
            "delta_y": result.delta_y,
            "next_y": result.next_y,
            "hard_active_species": result.active_species,
        }


def _patankar_candidates(
    current_y: torch.Tensor,
    demand: torch.Tensor,
    consumption_matrix,
    *,
    floor: float,
    numerical_tolerance: float,
    availability_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    demand64 = demand.to(torch.float64)
    y0 = current_y.to(device=demand64.device, dtype=torch.float64)
    consumption = _float64_matrix(consumption_matrix, demand64)
    if consumption.layout != torch.strided:
        consumption = consumption.to_dense()
    if consumption.ndim != 2 or demand64.shape[1] != consumption.shape[1]:
        raise ValueError("consumption_matrix must have shape (species, processes)")
    if y0.shape != (demand64.shape[0], consumption.shape[0]):
        raise ValueError("current_y shape does not match process consumption")
    requested = demand64 @ consumption.T
    margin = (y0 - float(floor) + float(numerical_tolerance)).clamp_min(0.0)
    availability = torch.clamp(
        margin / requested.clamp_min(float(availability_floor)),
        min=0.0,
        max=1.0,
    )
    reactant_mask = consumption.T > 0.0
    candidates = torch.where(
        reactant_mask[None],
        availability[:, None, :],
        torch.ones((), dtype=torch.float64, device=availability.device),
    )
    hard_scale, active = torch.min(candidates, dim=-1)
    active = torch.where(
        reactant_mask.any(dim=-1)[None],
        active,
        torch.full_like(active, -1),
    )
    return demand64, requested, availability, reactant_mask, hard_scale, active


def hard_patankar_process_availability(
    current_y: torch.Tensor,
    demand: torch.Tensor,
    consumption_matrix,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
    availability_floor: float = 1e-30,
) -> PatankarProcessAvailabilityResult:
    """Reproduce NeuralPatankar's hard reactant-availability ``amin``."""

    values = _patankar_candidates(
        current_y,
        demand,
        consumption_matrix,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
        availability_floor=availability_floor,
    )
    demand64, requested, availability, _mask, hard_scale, active = values
    return PatankarProcessAvailabilityResult(
        demand=demand64,
        requested_species=requested,
        species_availability=availability,
        process_scale=hard_scale,
        process_extent=demand64 * hard_scale,
        active_species=active,
    )


def smooth_safe_patankar_process_availability(
    current_y: torch.Tensor,
    demand: torch.Tensor,
    consumption_matrix,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
    availability_floor: float = 1e-30,
    p_norm: float = 32.0,
    roundoff_safety_ulps: float = 8.0,
) -> PatankarProcessAvailabilityResult:
    r"""Replace Patankar's reactant ``amin`` by a safe smooth lower bound.

    For reactant availabilities ``a_ji``, the generalized harmonic minimum
    ``a_j = (sum_i a_ji**(-p))**(-1/p)`` is no larger than any participating
    ``a_ji``. Total resource consumption therefore remains bounded by the
    current inventory, while reactant ties have a continuous derivative.
    """

    if p_norm < 1.0:
        raise ValueError("p_norm must be at least one")
    values = _patankar_candidates(
        current_y,
        demand,
        consumption_matrix,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
        availability_floor=availability_floor,
    )
    demand64, requested, availability, reactant_mask, hard_scale, active = values
    inverse_log = -torch.log(availability.clamp_min(torch.finfo(torch.float64).tiny))
    terms = float(p_norm) * inverse_log[:, None, :]
    terms = torch.where(
        reactant_mask[None],
        terms,
        torch.full_like(terms, -torch.inf),
    )
    has_reactant = reactant_mask.any(dim=-1)[None]
    log_upper_inverse = torch.logsumexp(terms, dim=-1) / float(p_norm)
    scale = torch.where(
        has_reactant,
        torch.exp(-log_upper_inverse),
        torch.ones_like(log_upper_inverse),
    )
    safe_scale = scale * (
        1.0 - float(roundoff_safety_ulps) * torch.finfo(torch.float64).eps
    )
    scale = torch.where(has_reactant, safe_scale, torch.ones_like(safe_scale))
    scale = torch.where(scale > hard_scale, hard_scale, scale).clamp(0.0, 1.0)
    return PatankarProcessAvailabilityResult(
        demand=demand64,
        requested_species=requested,
        species_availability=availability,
        process_scale=scale,
        process_extent=demand64 * scale,
        active_species=active,
    )


def scaling_diagnostics(
    result: ReactionExtentScalingResult,
    *,
    floor: float = 0.0,
    activation_tolerance: float = 1e-12,
) -> dict[str, float]:
    """Summarize activation, correction size, and feasibility."""

    extent_correction = result.scaled_extent - result.proposed_extent
    delta_correction = result.delta_y - result.proposed_delta_y
    extent_norm = torch.linalg.vector_norm(result.proposed_extent, dim=-1).clamp_min(1e-300)
    delta_norm = torch.linalg.vector_norm(result.proposed_delta_y, dim=-1).clamp_min(1e-300)
    violation = (float(floor) - result.next_y).clamp_min(0.0)
    return {
        "activation_rate": float(
            (result.scale[:, 0] < 1.0 - activation_tolerance).to(torch.float64).mean().detach().cpu()
        ),
        "hard_active_species_rate": float(
            (result.active_species >= 0).to(torch.float64).mean().detach().cpu()
        ),
        "mean_scale": float(result.scale.mean().detach().cpu()),
        "minimum_scale": float(result.scale.min().detach().cpu()),
        "mean_absolute_scale_correction": float((1.0 - result.scale).mean().detach().cpu()),
        "mean_relative_extent_correction_l2": float(
            (torch.linalg.vector_norm(extent_correction, dim=-1) / extent_norm).mean().detach().cpu()
        ),
        "mean_relative_delta_y_correction_l2": float(
            (torch.linalg.vector_norm(delta_correction, dim=-1) / delta_norm).mean().detach().cpu()
        ),
        "minimum_next_y": float(result.next_y.min().detach().cpu()),
        "maximum_positivity_violation": float(violation.max().detach().cpu()),
    }


def deployment_limiter_diagnostics(
    current_y: torch.Tensor,
    model_delta_y: torch.Tensor,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
    material_tolerance: float = 1e-12,
) -> dict[str, float]:
    """Measure whether a second UDF-style global limiter changes the output."""

    cap, _active = hard_fraction_to_boundary(
        current_y,
        model_delta_y,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
    )
    delta64 = model_delta_y.to(torch.float64)
    corrected = cap * delta64
    raw_next = current_y.to(torch.float64) + delta64
    corrected_next = current_y.to(torch.float64) + corrected
    correction_norm = torch.linalg.vector_norm(corrected - delta64, dim=-1)
    source_norm = torch.linalg.vector_norm(delta64, dim=-1).clamp_min(1e-300)
    return {
        "exact_activation_rate": float((cap[:, 0] < 1.0).to(torch.float64).mean().detach().cpu()),
        "material_activation_rate": float(
            (cap[:, 0] < 1.0 - material_tolerance).to(torch.float64).mean().detach().cpu()
        ),
        "mean_scale": float(cap.mean().detach().cpu()),
        "minimum_scale": float(cap.min().detach().cpu()),
        "mean_relative_delta_correction_l2": float(
            (correction_norm / source_norm).mean().detach().cpu()
        ),
        "minimum_raw_next_y": float(raw_next.min().detach().cpu()),
        "minimum_corrected_next_y": float(corrected_next.min().detach().cpu()),
    }


def compare_hard_and_smooth_scaling(
    current_y: torch.Tensor,
    proposed_extent: torch.Tensor,
    stoichiometric_mass_matrix,
    *,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
    p_norm: float = 32.0,
    transition_width: float = 1e-3,
) -> tuple[ReactionExtentScalingResult, ReactionExtentScalingResult, dict[str, float]]:
    """Evaluate model-level and second deployment limiters together."""

    hard = hard_reaction_extent_scaling(
        current_y,
        proposed_extent,
        stoichiometric_mass_matrix,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
    )
    smooth = smooth_safe_reaction_extent_scaling(
        current_y,
        proposed_extent,
        stoichiometric_mass_matrix,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
        p_norm=p_norm,
        transition_width=transition_width,
    )
    metrics = {
        f"hard_{key}": value
        for key, value in scaling_diagnostics(hard, floor=floor).items()
    }
    metrics.update(
        {
            f"smooth_{key}": value
            for key, value in scaling_diagnostics(smooth, floor=floor).items()
        }
    )
    metrics.update(
        {
            f"smooth_post_deployment_{key}": value
            for key, value in deployment_limiter_diagnostics(
                current_y, smooth.delta_y, floor=floor, numerical_tolerance=0.0
            ).items()
        }
    )
    metrics["mean_smooth_to_hard_scale_gap"] = float(
        (hard.scale - smooth.scale).mean().detach().cpu()
    )
    metrics["maximum_smooth_minus_hard_scale"] = float(
        (smooth.scale - hard.scale).max().detach().cpu()
    )
    return hard, smooth, metrics


def _one_sided_jvp(base, plus, minus, epsilon: float):
    forward = (plus - base) / epsilon
    backward = (base - minus) / epsilon
    jump = torch.linalg.vector_norm(forward - backward, dim=-1)
    reference = 0.5 * (
        torch.linalg.vector_norm(forward, dim=-1)
        + torch.linalg.vector_norm(backward, dim=-1)
    )
    return jump, jump / reference.clamp_min(1e-300)


def directional_smoothness_diagnostics(
    current_y: torch.Tensor,
    proposed_extent: torch.Tensor,
    stoichiometric_mass_matrix,
    current_y_direction: torch.Tensor,
    *,
    reaction_extent_direction: torch.Tensor | None = None,
    epsilon: float = 1e-7,
    floor: float = 0.0,
    numerical_tolerance: float = 0.0,
    p_norm: float = 32.0,
    transition_width: float = 1e-3,
) -> dict[str, float]:
    """Compare one-sided source JVPs across a Fluent-like perturbation."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    y0 = current_y.to(torch.float64)
    xi0 = proposed_extent.to(torch.float64)
    y_direction = current_y_direction.to(device=y0.device, dtype=torch.float64)
    if y_direction.shape != y0.shape:
        raise ValueError("current_y_direction must match current_y")
    if reaction_extent_direction is None:
        extent_direction = torch.zeros_like(xi0)
    else:
        extent_direction = reaction_extent_direction.to(device=xi0.device, dtype=torch.float64)
        if extent_direction.shape != xi0.shape:
            raise ValueError("reaction_extent_direction must match proposed_extent")
    y_plus, y_minus = y0 + epsilon * y_direction, y0 - epsilon * y_direction
    xi_plus, xi_minus = xi0 + epsilon * extent_direction, xi0 - epsilon * extent_direction

    hard_kwargs = dict(
        stoichiometric_mass_matrix=stoichiometric_mass_matrix,
        floor=floor,
        numerical_tolerance=numerical_tolerance,
    )
    smooth_kwargs = dict(
        **hard_kwargs, p_norm=p_norm, transition_width=transition_width
    )
    hard_base = hard_reaction_extent_scaling(y0, xi0, **hard_kwargs)
    hard_plus = hard_reaction_extent_scaling(y_plus, xi_plus, **hard_kwargs)
    hard_minus = hard_reaction_extent_scaling(y_minus, xi_minus, **hard_kwargs)
    smooth_base = smooth_safe_reaction_extent_scaling(y0, xi0, **smooth_kwargs)
    smooth_plus = smooth_safe_reaction_extent_scaling(y_plus, xi_plus, **smooth_kwargs)
    smooth_minus = smooth_safe_reaction_extent_scaling(y_minus, xi_minus, **smooth_kwargs)
    hard_jump, hard_relative = _one_sided_jvp(
        hard_base.delta_y, hard_plus.delta_y, hard_minus.delta_y, epsilon
    )
    smooth_jump, smooth_relative = _one_sided_jvp(
        smooth_base.delta_y, smooth_plus.delta_y, smooth_minus.delta_y, epsilon
    )
    switch = (hard_plus.active_species != hard_minus.active_species) & (
        (hard_plus.active_species >= 0) | (hard_minus.active_species >= 0)
    )
    return {
        "hard_active_species_switch_rate": float(switch.to(torch.float64).mean().detach().cpu()),
        "hard_directional_jvp_jump_l2_mean": float(hard_jump.mean().detach().cpu()),
        "hard_directional_jvp_jump_l2_max": float(hard_jump.max().detach().cpu()),
        "hard_directional_jvp_relative_jump_mean": float(hard_relative.mean().detach().cpu()),
        "smooth_directional_jvp_jump_l2_mean": float(smooth_jump.mean().detach().cpu()),
        "smooth_directional_jvp_jump_l2_max": float(smooth_jump.max().detach().cpu()),
        "smooth_directional_jvp_relative_jump_mean": float(smooth_relative.mean().detach().cpu()),
        "smooth_to_hard_jvp_jump_ratio": float(
            (smooth_jump.mean() / hard_jump.mean().clamp_min(1e-300)).detach().cpu()
        ),
    }


def patankar_directional_smoothness_diagnostics(
    current_y: torch.Tensor,
    demand: torch.Tensor,
    consumption_matrix,
    current_y_direction: torch.Tensor,
    *,
    demand_direction: torch.Tensor | None = None,
    epsilon: float = 1e-7,
    p_norm: float = 32.0,
) -> dict[str, float]:
    """Measure active-reactant switching in NeuralPatankar allocation."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    y0, demand0 = current_y.to(torch.float64), demand.to(torch.float64)
    y_direction = current_y_direction.to(device=y0.device, dtype=torch.float64)
    demand_direction64 = (
        torch.zeros_like(demand0)
        if demand_direction is None
        else demand_direction.to(device=demand0.device, dtype=torch.float64)
    )
    y_plus, y_minus = y0 + epsilon * y_direction, y0 - epsilon * y_direction
    demand_plus = demand0 + epsilon * demand_direction64
    demand_minus = demand0 - epsilon * demand_direction64
    hard_base = hard_patankar_process_availability(y0, demand0, consumption_matrix)
    hard_plus = hard_patankar_process_availability(y_plus, demand_plus, consumption_matrix)
    hard_minus = hard_patankar_process_availability(y_minus, demand_minus, consumption_matrix)
    smooth_base = smooth_safe_patankar_process_availability(
        y0, demand0, consumption_matrix, p_norm=p_norm
    )
    smooth_plus = smooth_safe_patankar_process_availability(
        y_plus, demand_plus, consumption_matrix, p_norm=p_norm
    )
    smooth_minus = smooth_safe_patankar_process_availability(
        y_minus, demand_minus, consumption_matrix, p_norm=p_norm
    )
    hard_jump, _ = _one_sided_jvp(
        hard_base.process_extent,
        hard_plus.process_extent,
        hard_minus.process_extent,
        epsilon,
    )
    smooth_jump, _ = _one_sided_jvp(
        smooth_base.process_extent,
        smooth_plus.process_extent,
        smooth_minus.process_extent,
        epsilon,
    )
    switch = hard_plus.active_species != hard_minus.active_species
    return {
        "hard_active_reactant_switch_rate": float(switch.to(torch.float64).mean().detach().cpu()),
        "hard_process_jvp_jump_l2_mean": float(hard_jump.mean().detach().cpu()),
        "smooth_process_jvp_jump_l2_mean": float(smooth_jump.mean().detach().cpu()),
        "smooth_to_hard_process_jvp_jump_ratio": float(
            (smooth_jump.mean() / hard_jump.mean().clamp_min(1e-300)).detach().cpu()
        ),
        "hard_mean_process_scale": float(hard_base.process_scale.mean().detach().cpu()),
        "smooth_mean_process_scale": float(smooth_base.process_scale.mean().detach().cpu()),
    }
