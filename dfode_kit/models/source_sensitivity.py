"""Diagonal neural source sensitivities for Fluent source linearization.

Fluent ``dS[eqn]`` consumes a diagonal derivative for one transported
equation. The utilities here intentionally produce only
``dS_i / dY_i``. They neither construct nor approximate the full cross-species
chemistry Jacobian.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _as_float64(value, *, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if device is None:
            device = value.device
        return value.to(device=device, dtype=torch.float64)
    return torch.as_tensor(value, device=device, dtype=torch.float64)


def _batch_column(value, batch_size: int, name: str, *, device) -> torch.Tensor:
    result = _as_float64(value, device=device)
    if result.ndim == 0:
        result = result.expand(batch_size)
    elif result.shape == (batch_size, 1):
        result = result[:, 0]
    elif result.shape != (batch_size,):
        raise ValueError(f"{name} must be scalar, (batch,), or (batch, 1)")
    return result[:, None]


def endpoint_diagonal_to_source_diagonal(
    endpoint_diagonal,
    density,
    dt,
    *,
    delta_y=None,
    density_diagonal=None,
) -> torch.Tensor:
    """Convert endpoint diagonal sensitivity to raw source sensitivity.

    For ``S_i = rho (Yhat_i - Y_i) / dt``, this evaluates

    ``dS_i/dY_i = rho/dt (dYhat_i/dY_i - 1)``

    and optionally adds

    ``(drho/dY_i) delta_Y_i/dt``.

    The result is not yet constrained to be stabilizing; pass it through
    :class:`StabilizingDiagonalSourceAdapter` before returning it to Fluent.
    """

    endpoint = _as_float64(endpoint_diagonal)
    if endpoint.ndim != 2:
        raise ValueError("endpoint_diagonal must have shape (batch, n_species)")
    batch_size = endpoint.shape[0]
    rho = _batch_column(density, batch_size, "density", device=endpoint.device)
    interval = _batch_column(dt, batch_size, "dt", device=endpoint.device)
    if bool(torch.any(rho <= 0.0).detach().cpu()):
        raise ValueError("density must be positive")
    if bool(torch.any(interval <= 0.0).detach().cpu()):
        raise ValueError("dt must be positive")
    derivative = rho / interval * (endpoint - 1.0)
    if (delta_y is None) != (density_diagonal is None):
        raise ValueError("delta_y and density_diagonal must be supplied together")
    if delta_y is not None:
        increment = _as_float64(delta_y, device=endpoint.device)
        density_derivative = _as_float64(
            density_diagonal, device=endpoint.device
        )
        if increment.shape != endpoint.shape or density_derivative.shape != endpoint.shape:
            raise ValueError(
                "delta_y and density_diagonal must match endpoint_diagonal shape"
            )
        derivative = derivative + density_derivative * increment / interval
    return derivative


class StabilizingDiagonalSourceAdapter(nn.Module):
    """Project raw diagonal source derivatives into ``[-d_max, 0]``.

    ``hard-negative`` exactly clips positive derivatives to zero and limits the
    most negative value. ``smooth-negative`` uses a softplus negative-part and
    tanh magnitude cap; it is differentiable but introduces a small negative
    value near a raw derivative of zero controlled by ``smoothness``.
    """

    diagonal_only = True
    full_jacobian = False

    def __init__(
        self,
        max_abs_derivative: float,
        *,
        policy: str = "hard-negative",
        smoothness: float = 1.0,
    ) -> None:
        super().__init__()
        if max_abs_derivative <= 0.0:
            raise ValueError("max_abs_derivative must be positive")
        if policy not in {"hard-negative", "smooth-negative"}:
            raise ValueError("policy must be 'hard-negative' or 'smooth-negative'")
        if smoothness <= 0.0:
            raise ValueError("smoothness must be positive")
        self.max_abs_derivative = float(max_abs_derivative)
        self.policy = policy
        self.smoothness = float(smoothness)

    def forward(self, raw_diagonal) -> torch.Tensor:
        raw = _as_float64(raw_diagonal)
        if raw.ndim != 2:
            raise ValueError("raw_diagonal must have shape (batch, n_species)")
        if self.policy == "hard-negative":
            return torch.clamp(
                raw, min=-self.max_abs_derivative, max=0.0
            )
        negative_magnitude = self.smoothness * F.softplus(
            -raw / self.smoothness
        )
        return -self.max_abs_derivative * torch.tanh(
            negative_magnitude / self.max_abs_derivative
        )


class CompactDiagonalSourceDerivativeHead(nn.Module):
    """Compact head for bounded Fluent ``dS[eqn]`` values.

    The neural trunk runs at its parameter dtype, normally float32. The output
    is converted to float64 and mapped smoothly to ``[-d_max_i, 0]`` through
    ``-d_max_i * sigmoid(logit_i)``. This is a learned diagonal preconditioner,
    not a full source Jacobian.
    """

    diagonal_only = True
    full_jacobian = False

    def __init__(
        self,
        input_dim: int,
        n_species: int,
        *,
        hidden_dim: int = 32,
        bottleneck_dim: int = 8,
        max_abs_derivative=1.0,
        initial_fraction: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if min(input_dim, n_species, hidden_dim, bottleneck_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if not 0.0 < initial_fraction < 1.0:
            raise ValueError("initial_fraction must lie strictly between 0 and 1")
        maximum = _as_float64(max_abs_derivative)
        if maximum.ndim == 0:
            maximum = maximum.expand(n_species).clone()
        if maximum.shape != (n_species,):
            raise ValueError("max_abs_derivative must be scalar or (n_species,)")
        if bool(torch.any(maximum <= 0.0).detach().cpu()):
            raise ValueError("max_abs_derivative must be positive")
        self.register_buffer("max_abs_derivative", maximum)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, n_species),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        initial_logit = math.log(initial_fraction / (1.0 - initial_fraction))
        nn.init.constant_(final.bias, initial_logit)

    def forward(self, features) -> dict[str, torch.Tensor]:
        parameter = next(self.network.parameters())
        if isinstance(features, torch.Tensor):
            feature_tensor = features.to(
                device=parameter.device, dtype=parameter.dtype
            )
        else:
            feature_tensor = torch.as_tensor(
                features, device=parameter.device, dtype=parameter.dtype
            )
        if feature_tensor.ndim != 2:
            raise ValueError("features must have shape (batch, input_dim)")
        raw_logits = self.network(feature_tensor)
        fraction = torch.sigmoid(raw_logits).to(torch.float64)
        derivative = -fraction * self.max_abs_derivative.to(
            device=fraction.device
        )
        return {
            "source_derivative_diagonal": derivative,
            "raw_logits": raw_logits,
            "bounded_fraction": fraction,
        }
