from __future__ import annotations

import torch


def signed_power(value: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.sign(value) * torch.abs(value).pow(alpha) / alpha


def signed_power_inverse(value: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.sign(value) * (alpha * torch.abs(value)).pow(1.0 / alpha)


class ResidualBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
        )
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ConservedIntervalModel(torch.nn.Module):
    """Predict an identifiable conserved update in completion coordinates."""

    def __init__(
        self,
        input_dim: int,
        completion_matrix,
        target_mean,
        target_std,
        *,
        hidden_dim: int = 256,
        depth: int = 4,
        target_alpha: float = 0.1,
        positivity_floor: float = 0.0,
    ):
        super().__init__()
        completion = torch.as_tensor(completion_matrix, dtype=torch.float64)
        target_mean = torch.as_tensor(target_mean, dtype=torch.float32)
        target_std = torch.as_tensor(target_std, dtype=torch.float32)
        if completion.ndim != 2:
            raise ValueError("completion_matrix must be two-dimensional")
        if target_mean.shape != (completion.shape[1],):
            raise ValueError("target statistics do not match completion coordinates")
        self.register_buffer("completion_matrix", completion)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.target_alpha = float(target_alpha)
        self.positivity_floor = float(positivity_floor)
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
        ]
        layers.extend(ResidualBlock(hidden_dim) for _ in range(depth))
        layers.append(torch.nn.Linear(hidden_dim, completion.shape[1]))
        self.network = torch.nn.Sequential(*layers)
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor, current_y: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_coordinates = self.network(features)
        transformed_coordinates = raw_coordinates * self.target_std + self.target_mean
        coordinates = signed_power_inverse(
            transformed_coordinates,
            self.target_alpha,
        ).to(torch.float64)
        unconstrained_delta = coordinates @ self.completion_matrix.T

        current_y64 = current_y.to(torch.float64)
        negative = unconstrained_delta < 0.0
        available = torch.clamp(current_y64 - self.positivity_floor, min=0.0)
        ratios = torch.where(
            negative,
            available / torch.clamp(-unconstrained_delta, min=torch.finfo(torch.float64).tiny),
            torch.full_like(unconstrained_delta, torch.inf),
        )
        limiter = torch.minimum(
            torch.ones((features.shape[0], 1), dtype=torch.float64, device=features.device),
            torch.amin(ratios, dim=-1, keepdim=True),
        ).clamp(min=0.0)
        delta_y = limiter * unconstrained_delta
        next_y = current_y64 + delta_y
        return {
            "raw_coordinates": raw_coordinates,
            "coordinates": coordinates,
            "unconstrained_delta_y": unconstrained_delta,
            "limiter": limiter,
            "delta_y": delta_y,
            "next_y": next_y,
        }
