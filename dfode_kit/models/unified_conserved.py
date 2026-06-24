from __future__ import annotations

import re

import torch

from dfode_kit.models.latent_baseline import LatentTimeEventHead, signed_power_inverse


def safe_module_key(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    if not safe or safe[0].isdigit():
        safe = f"m_{safe}"
    return safe


class SharedLatentDynamicsConservedModel(torch.nn.Module):
    """Mechanism-specific conserved heads with a shared latent dynamics trunk."""

    def __init__(
        self,
        mechanism_configs: dict[str, dict],
        *,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        transform_alpha: float = 0.1,
        transform_scale_by_alpha: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.transform_alpha = transform_alpha
        self.transform_scale_by_alpha = transform_scale_by_alpha
        self.safe_keys = {name: safe_module_key(name) for name in mechanism_configs}
        self.mechanism_names = list(mechanism_configs)

        self.encoders = torch.nn.ModuleDict()
        self.delta_key_heads = torch.nn.ModuleDict()
        self.temperature_pressure_heads = torch.nn.ModuleDict()
        self.time_event_heads = torch.nn.ModuleDict()

        self.shared_dynamics = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )

        for name, cfg in mechanism_configs.items():
            key = self.safe_keys[name]
            input_dim = int(cfg["input_dim"])
            completion = torch.as_tensor(cfg["completion_matrix"], dtype=torch.float64)
            if completion.ndim != 2:
                raise ValueError(f"completion_matrix for {name} must be 2D")
            n_species, n_key = completion.shape
            if input_dim != 2 + n_species:
                raise ValueError(f"input_dim for {name} is inconsistent with completion matrix")
            self.register_buffer(f"completion_matrix__{key}", completion)
            self.encoders[key] = torch.nn.Sequential(
                torch.nn.Linear(input_dim + 1, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, latent_dim),
            )
            self.delta_key_heads[key] = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, n_key),
            )
            self.temperature_pressure_heads[key] = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, 2),
            )
            self.time_event_heads[key] = LatentTimeEventHead(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def completion_matrix(self, mechanism_name: str):
        return getattr(self, f"completion_matrix__{self.safe_keys[mechanism_name]}")

    def forward(self, mechanism_name: str, x_current, log_dt, current_species=None):
        key = self.safe_keys[mechanism_name]
        z = self.encoders[key](torch.cat([x_current, log_dt], dim=-1))
        z_next = self.shared_dynamics(z)
        raw_key = self.delta_key_heads[key](z_next)
        delta_key = signed_power_inverse(
            raw_key,
            alpha=self.transform_alpha,
            scale_by_alpha=self.transform_scale_by_alpha,
        )
        delta_y = delta_key.to(torch.float64) @ self.completion_matrix(mechanism_name).T
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)
        next_tp = self.temperature_pressure_heads[key](z_next).to(torch.float64)
        next_y = current_species + delta_y
        time_event = self.time_event_heads[key](z_next)
        return {
            "next_state": torch.cat([next_tp, next_y], dim=-1),
            "raw_key": raw_key,
            "delta_key": delta_key,
            "delta_y": delta_y,
            "z": z,
            "z_next": z_next,
            "log_dt": time_event["log_dt"],
            "equilibrium_logit": time_event["equilibrium_logit"],
        }
