"""Deployment-oriented losses for Fluent-native direct-integration labels.

The functions in this module deliberately do not create chemistry labels. All
target endpoints must come from Ansys Fluent native direct integration. Neural
predictions may remain float32 internally, but every physical quantity formed
here is evaluated in float64.

No loss scale is estimated from the batch. Each term has an explicit physical
scale and weight, and the returned report exposes the raw, scaled, and weighted
values separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


FLUENT_NATIVE_DI_BACKEND = "ansys-fluent-native-di"

SpeciesEnthalpyCallback = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LossTerm:
    """Explicit normalization and weight for one physical loss term."""

    weight: float
    scale: float

    def __post_init__(self) -> None:
        if not float(self.weight) >= 0.0:
            raise ValueError("loss weight must be non-negative")
        if not float(self.scale) > 0.0:
            raise ValueError("loss scale must be positive")


@dataclass(frozen=True)
class FluentDeploymentLossConfig:
    """Configuration with no batch-derived or hidden normalization."""

    source: LossTerm
    mixture_molecular_weight: LossTerm
    density_increment: LossTerm
    enthalpy: LossTerm
    heat_release: LossTerm


@dataclass(frozen=True)
class FluentJVPConsistencyConfig:
    """Explicit scales and weights for endpoint and source directional JVPs."""

    endpoint: LossTerm
    source: LossTerm


def _require_fluent_native_di(label_backend: str) -> None:
    if label_backend != FLUENT_NATIVE_DI_BACKEND:
        raise ValueError(
            "deployment targets must use Ansys Fluent native direct-integration "
            f"labels ({FLUENT_NATIVE_DI_BACKEND!r}); received {label_backend!r}"
        )


def _as_float64(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is None:
            device = value.device
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(value, device=device, dtype=torch.float64)


def _as_batch_vector(value, batch_size: int, name: str, *, device) -> torch.Tensor:
    result = _as_float64(value, device=device)
    if result.ndim == 0:
        result = result.expand(batch_size)
    elif result.shape == (batch_size, 1):
        result = result[:, 0]
    elif result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar, (batch,), or (batch, 1)")
    return result


def _validate_state_matrix(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape (batch, features)")


def _expand_species_values(
    value,
    batch_size: int,
    n_species: int,
    name: str,
    *,
    device,
) -> torch.Tensor:
    result = _as_float64(value, device=device)
    if result.shape == (n_species,):
        result = result.unsqueeze(0).expand(batch_size, -1)
    elif result.shape != (batch_size, n_species):
        raise ValueError(
            f"{name} must have shape (n_species,) or (batch, n_species)"
        )
    return result


def mixture_molecular_weight(
    mass_fractions: torch.Tensor,
    molecular_weights: torch.Tensor,
) -> torch.Tensor:
    """Return ``1 / sum_i(Y_i / W_i)`` without renormalizing composition."""

    y = _as_float64(mass_fractions)
    _validate_state_matrix(y, "mass_fractions")
    weights = _as_float64(molecular_weights, device=y.device)
    if weights.shape != (y.shape[1],):
        raise ValueError("molecular_weights must have shape (n_species,)")
    if bool(torch.any(weights <= 0.0).detach().cpu()):
        raise ValueError("molecular_weights must be positive")
    return torch.reciprocal(torch.sum(y / weights, dim=-1))


def ideal_gas_density(
    pressure,
    temperature,
    mass_fractions: torch.Tensor,
    molecular_weights: torch.Tensor,
    *,
    gas_constant: float,
) -> torch.Tensor:
    """Evaluate ``rho = p W_mix / (R T)`` in caller-selected consistent units."""

    y = _as_float64(mass_fractions)
    _validate_state_matrix(y, "mass_fractions")
    batch_size = y.shape[0]
    p = _as_batch_vector(pressure, batch_size, "pressure", device=y.device)
    temperature64 = _as_batch_vector(
        temperature, batch_size, "temperature", device=y.device
    )
    if float(gas_constant) <= 0.0:
        raise ValueError("gas_constant must be positive")
    if bool(torch.any(temperature64 <= 0.0).detach().cpu()):
        raise ValueError("temperature must be positive")
    return (
        p
        * mixture_molecular_weight(y, molecular_weights)
        / (float(gas_constant) * temperature64)
    )


def mixture_specific_enthalpy(
    mass_fractions: torch.Tensor,
    species_specific_enthalpy: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``h = sum_i Y_i h_i`` without composition correction."""

    y = _as_float64(mass_fractions)
    _validate_state_matrix(y, "mass_fractions")
    h_species = _expand_species_values(
        species_specific_enthalpy,
        y.shape[0],
        y.shape[1],
        "species_specific_enthalpy",
        device=y.device,
    )
    return torch.sum(y * h_species, dim=-1)


def specific_species_source(delta_y: torch.Tensor, dt) -> torch.Tensor:
    """Convert an endpoint mass-fraction increment to ``delta_Y / dt``."""

    increment = _as_float64(delta_y)
    _validate_state_matrix(increment, "delta_y")
    interval = _as_batch_vector(
        dt, increment.shape[0], "dt", device=increment.device
    )
    if bool(torch.any(interval <= 0.0).detach().cpu()):
        raise ValueError("dt must be positive")
    return increment / interval[:, None]


def volumetric_heat_release(
    delta_y: torch.Tensor,
    dt,
    species_specific_enthalpy: torch.Tensor,
    density,
) -> torch.Tensor:
    """Evaluate ``qdot = -rho sum_i(h_i delta_Y_i / dt)``."""

    source = specific_species_source(delta_y, dt)
    h_species = _expand_species_values(
        species_specific_enthalpy,
        source.shape[0],
        source.shape[1],
        "species_specific_enthalpy",
        device=source.device,
    )
    rho = _as_batch_vector(
        density, source.shape[0], "density", device=source.device
    )
    return -rho * torch.sum(h_species * source, dim=-1)


def _species_enthalpy_at(
    callback: SpeciesEnthalpyCallback,
    temperature: torch.Tensor,
    *,
    batch_size: int,
    n_species: int,
    device,
) -> torch.Tensor:
    values = callback(temperature)
    return _expand_species_values(
        values,
        batch_size,
        n_species,
        "species_enthalpy callback output",
        device=device,
    )


def _loss_report(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: LossTerm,
) -> dict[str, torch.Tensor]:
    raw_mae = torch.mean(torch.abs(prediction - target))
    scale = torch.as_tensor(config.scale, dtype=torch.float64, device=raw_mae.device)
    weight = torch.as_tensor(config.weight, dtype=torch.float64, device=raw_mae.device)
    scaled = raw_mae / scale
    contribution = weight * scaled
    return {
        "raw_mae": raw_mae,
        "scale": scale,
        "scaled": scaled,
        "weight": weight,
        "contribution": contribution,
    }


def fluent_deployment_losses(
    *,
    current_y,
    predicted_delta_y,
    target_delta_y,
    current_temperature,
    predicted_temperature,
    target_temperature,
    pressure,
    dt,
    molecular_weights,
    species_enthalpy: SpeciesEnthalpyCallback,
    config: FluentDeploymentLossConfig,
    label_backend: str,
    gas_constant: float = 8314.46261815324,
    heat_release_species_enthalpy=None,
) -> dict[str, object]:
    """Build differentiable deployment losses against Fluent-DI endpoints.

    ``species_enthalpy(T)`` must return species specific enthalpies with shape
    ``(batch, n_species)`` or ``(n_species,)``. The callback is supplied by the
    caller so training can use thermodynamics compatible with the deployment
    contract. If ``heat_release_species_enthalpy`` is omitted, species
    enthalpies evaluated at the current temperature are used for both the
    prediction and Fluent-DI target.
    """

    _require_fluent_native_di(label_backend)
    predicted_increment = _as_float64(predicted_delta_y)
    _validate_state_matrix(predicted_increment, "predicted_delta_y")
    device = predicted_increment.device
    current = _as_float64(current_y, device=device)
    target_increment = _as_float64(target_delta_y, device=device)
    if current.shape != predicted_increment.shape or current.shape != target_increment.shape:
        raise ValueError("current_y and both delta_y tensors must have identical shapes")

    batch_size, n_species = current.shape
    current_t = _as_batch_vector(
        current_temperature, batch_size, "current_temperature", device=device
    )
    predicted_t = _as_batch_vector(
        predicted_temperature, batch_size, "predicted_temperature", device=device
    )
    target_t = _as_batch_vector(
        target_temperature, batch_size, "target_temperature", device=device
    )
    interval = _as_batch_vector(dt, batch_size, "dt", device=device)
    if bool(torch.any(interval <= 0.0).detach().cpu()):
        raise ValueError("dt must be positive")
    weights = _as_float64(molecular_weights, device=device)

    predicted_y = current + predicted_increment
    target_y = current + target_increment
    predicted_source = specific_species_source(predicted_increment, interval)
    target_source = specific_species_source(target_increment, interval)

    current_density = ideal_gas_density(
        pressure,
        current_t,
        current,
        weights,
        gas_constant=gas_constant,
    )
    predicted_density = ideal_gas_density(
        pressure,
        predicted_t,
        predicted_y,
        weights,
        gas_constant=gas_constant,
    )
    target_density = ideal_gas_density(
        pressure,
        target_t,
        target_y,
        weights,
        gas_constant=gas_constant,
    )
    predicted_density_increment = predicted_density - current_density
    target_density_increment = target_density - current_density

    current_h_species = _species_enthalpy_at(
        species_enthalpy,
        current_t,
        batch_size=batch_size,
        n_species=n_species,
        device=device,
    )
    predicted_h_species = _species_enthalpy_at(
        species_enthalpy,
        predicted_t,
        batch_size=batch_size,
        n_species=n_species,
        device=device,
    )
    target_h_species = _species_enthalpy_at(
        species_enthalpy,
        target_t,
        batch_size=batch_size,
        n_species=n_species,
        device=device,
    )
    predicted_enthalpy = mixture_specific_enthalpy(
        predicted_y, predicted_h_species
    )
    target_enthalpy = mixture_specific_enthalpy(target_y, target_h_species)

    if heat_release_species_enthalpy is None:
        heat_release_h_species = current_h_species
    elif callable(heat_release_species_enthalpy):
        heat_release_h_species = _species_enthalpy_at(
            heat_release_species_enthalpy,
            current_t,
            batch_size=batch_size,
            n_species=n_species,
            device=device,
        )
    else:
        heat_release_h_species = _expand_species_values(
            heat_release_species_enthalpy,
            batch_size,
            n_species,
            "heat_release_species_enthalpy",
            device=device,
        )
    predicted_heat_release = volumetric_heat_release(
        predicted_increment,
        interval,
        heat_release_h_species,
        current_density,
    )
    target_heat_release = volumetric_heat_release(
        target_increment,
        interval,
        heat_release_h_species,
        current_density,
    )

    terms = {
        "source": _loss_report(predicted_source, target_source, config.source),
        "mixture_molecular_weight": _loss_report(
            mixture_molecular_weight(predicted_y, weights),
            mixture_molecular_weight(target_y, weights),
            config.mixture_molecular_weight,
        ),
        "density_increment": _loss_report(
            predicted_density_increment,
            target_density_increment,
            config.density_increment,
        ),
        "enthalpy": _loss_report(
            predicted_enthalpy, target_enthalpy, config.enthalpy
        ),
        "heat_release": _loss_report(
            predicted_heat_release, target_heat_release, config.heat_release
        ),
    }
    total = torch.stack([term["contribution"] for term in terms.values()]).sum()
    return {
        "total": total,
        "terms": terms,
        "derived": {
            "predicted_source": predicted_source,
            "target_source": target_source,
            "predicted_mixture_molecular_weight": mixture_molecular_weight(
                predicted_y, weights
            ),
            "target_mixture_molecular_weight": mixture_molecular_weight(
                target_y, weights
            ),
            "predicted_density_increment": predicted_density_increment,
            "target_density_increment": target_density_increment,
            "predicted_enthalpy": predicted_enthalpy,
            "target_enthalpy": target_enthalpy,
            "predicted_heat_release": predicted_heat_release,
            "target_heat_release": target_heat_release,
        },
        "label_backend": label_backend,
    }


def directional_finite_difference(
    base: torch.Tensor,
    perturbed: torch.Tensor,
    epsilon,
) -> torch.Tensor:
    """Return ``(f(x + epsilon v) - f(x)) / epsilon`` in float64."""

    base64 = _as_float64(base)
    perturbed64 = _as_float64(perturbed, device=base64.device)
    if base64.shape != perturbed64.shape or base64.ndim != 2:
        raise ValueError("base and perturbed values must have identical 2D shapes")
    step = _as_batch_vector(
        epsilon, base64.shape[0], "epsilon", device=base64.device
    )
    if bool(torch.any(step <= 0.0).detach().cpu()):
        raise ValueError("epsilon must be positive")
    return (perturbed64 - base64) / step[:, None]


def fluent_di_jvp_consistency_losses(
    *,
    current_state,
    perturbed_state,
    predicted_endpoint,
    predicted_perturbed_endpoint,
    target_di_endpoint,
    target_di_perturbed_endpoint,
    epsilon,
    dt,
    config: FluentJVPConsistencyConfig,
    label_backend: str,
    species_start: int = 2,
) -> dict[str, object]:
    """Compare endpoint and source JVPs from paired Fluent-DI states.

    The source JVP uses

    ``S_Y(x) = (Phi_Y(x) - Y) / dt``.

    Therefore, the perturbed source subtracts the perturbed input composition,
    not the unperturbed one. This function learns a directional action only; it
    does not reconstruct or claim a full chemical Jacobian.
    """

    _require_fluent_native_di(label_backend)
    predicted = _as_float64(predicted_endpoint)
    _validate_state_matrix(predicted, "predicted_endpoint")
    device = predicted.device
    tensors = {
        "current_state": _as_float64(current_state, device=device),
        "perturbed_state": _as_float64(perturbed_state, device=device),
        "predicted_perturbed_endpoint": _as_float64(
            predicted_perturbed_endpoint, device=device
        ),
        "target_di_endpoint": _as_float64(target_di_endpoint, device=device),
        "target_di_perturbed_endpoint": _as_float64(
            target_di_perturbed_endpoint, device=device
        ),
    }
    for name, value in tensors.items():
        if value.shape != predicted.shape:
            raise ValueError(f"{name} must match predicted_endpoint shape")
    if not 0 <= species_start < predicted.shape[1]:
        raise ValueError("species_start must select at least one endpoint column")

    predicted_endpoint_jvp = directional_finite_difference(
        predicted, tensors["predicted_perturbed_endpoint"], epsilon
    )
    target_endpoint_jvp = directional_finite_difference(
        tensors["target_di_endpoint"],
        tensors["target_di_perturbed_endpoint"],
        epsilon,
    )

    interval = _as_batch_vector(dt, predicted.shape[0], "dt", device=device)
    if bool(torch.any(interval <= 0.0).detach().cpu()):
        raise ValueError("dt must be positive")
    current_y = tensors["current_state"][:, species_start:]
    perturbed_y = tensors["perturbed_state"][:, species_start:]
    predicted_source = (
        predicted[:, species_start:] - current_y
    ) / interval[:, None]
    predicted_perturbed_source = (
        tensors["predicted_perturbed_endpoint"][:, species_start:] - perturbed_y
    ) / interval[:, None]
    target_source = (
        tensors["target_di_endpoint"][:, species_start:] - current_y
    ) / interval[:, None]
    target_perturbed_source = (
        tensors["target_di_perturbed_endpoint"][:, species_start:] - perturbed_y
    ) / interval[:, None]
    predicted_source_jvp = directional_finite_difference(
        predicted_source, predicted_perturbed_source, epsilon
    )
    target_source_jvp = directional_finite_difference(
        target_source, target_perturbed_source, epsilon
    )

    terms = {
        "endpoint_jvp": _loss_report(
            predicted_endpoint_jvp, target_endpoint_jvp, config.endpoint
        ),
        "source_jvp": _loss_report(
            predicted_source_jvp, target_source_jvp, config.source
        ),
    }
    total = torch.stack([term["contribution"] for term in terms.values()]).sum()
    return {
        "total": total,
        "terms": terms,
        "derived": {
            "predicted_endpoint_jvp": predicted_endpoint_jvp,
            "target_endpoint_jvp": target_endpoint_jvp,
            "predicted_source_jvp": predicted_source_jvp,
            "target_source_jvp": target_source_jvp,
        },
        "label_backend": label_backend,
        "full_jacobian_reconstructed": False,
    }
