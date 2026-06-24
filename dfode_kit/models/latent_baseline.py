from __future__ import annotations

import torch


def signed_power_transform(value, *, alpha: float = 0.5, eps: float = 1e-12, scale_by_alpha: bool = False):
    abs_value = torch.abs(value)
    pure = torch.sign(value) * (abs_value**alpha)
    if scale_by_alpha:
        pure = pure / alpha
    if not value.requires_grad:
        return pure
    grad_floor = 1e-30 if value.dtype == torch.float64 else 1e-12
    safe = torch.sign(value) * (torch.clamp(abs_value, min=grad_floor) ** alpha)
    if scale_by_alpha:
        safe = safe / alpha
    return pure.detach() + safe - safe.detach()


def signed_power_inverse(value, *, alpha: float = 0.5, eps: float = 1e-12, scale_by_alpha: bool = False):
    if scale_by_alpha:
        value = value * alpha
    return torch.sign(value) * (torch.abs(value) ** (1.0 / alpha))


class StateAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class LatentGRURollout(torch.nn.Module):
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = torch.nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.project = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_sequence):
        hidden, _ = self.gru(z_sequence)
        return self.project(hidden)


class LatentTimeEventHead(torch.nn.Module):
    """Predict adaptive physical time advance and equilibrium event probability."""

    def __init__(self, latent_dim: int = 16, condition_dim: int = 0, hidden_dim: int = 128):
        super().__init__()
        input_dim = latent_dim + condition_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.log_dt = torch.nn.Linear(hidden_dim, 1)
        self.equilibrium_logit = torch.nn.Linear(hidden_dim, 1)

    def forward(self, z, condition=None):
        if condition is not None:
            z = torch.cat([z, condition], dim=-1)
        hidden = self.net(z)
        return {
            "log_dt": self.log_dt(hidden),
            "equilibrium_logit": self.equilibrium_logit(hidden),
        }


class AutoencoderLatentGRU(torch.nn.Module):
    """Minimal AE + GRU latent rollout baseline for sequence experiments."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 1,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.autoencoder = StateAutoencoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.rollout = LatentGRURollout(latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        self.time_event_head = LatentTimeEventHead(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
        )

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def reconstruct(self, x):
        return self.autoencoder(x)

    def forward(self, x_sequence):
        batch, steps, features = x_sequence.shape
        flat = x_sequence.reshape(batch * steps, features)
        z_flat = self.encode(flat)
        z_sequence = z_flat.reshape(batch, steps, -1)
        z_next = self.rollout(z_sequence[:, :-1, :])
        decoded_next = self.decode(z_next.reshape(batch * (steps - 1), -1))
        return decoded_next.reshape(batch, steps - 1, features), z_next

    def predict_time_event(self, z, condition=None):
        return self.time_event_head(z, condition=condition)


class ConservedLatentDeltaModel(torch.nn.Module):
    """Predict key-species deltas and complete atom-conserving species updates."""

    def __init__(
        self,
        input_dim: int,
        completion_matrix,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        key_mode: str = "direct",
        transform_alpha: float = 0.5,
        transform_eps: float = 1e-12,
        transform_scale_by_alpha: bool = False,
        key_species_indices=None,
        key_scale_alpha: float = 0.0,
    ):
        super().__init__()
        completion = torch.as_tensor(completion_matrix, dtype=torch.float64)
        if completion.ndim != 2:
            raise ValueError("completion_matrix must be 2D")
        n_species, n_key = completion.shape
        if input_dim != 2 + n_species:
            raise ValueError(f"input_dim {input_dim} is inconsistent with {n_species} species")

        self.input_dim = input_dim
        self.n_species = n_species
        self.n_key = n_key
        self.key_mode = key_mode
        self.transform_alpha = transform_alpha
        self.transform_eps = transform_eps
        self.transform_scale_by_alpha = transform_scale_by_alpha
        self.key_scale_alpha = key_scale_alpha
        if key_mode not in {"direct", "signed-power", "formation-consumption"}:
            raise ValueError(f"Unsupported key_mode: {key_mode}")
        self.register_buffer("completion_matrix", completion)
        if key_species_indices is None:
            key_species_indices = list(range(n_key))
        if len(key_species_indices) != n_key:
            raise ValueError("key_species_indices must have one entry per key dimension")
        self.register_buffer("key_species_indices", torch.as_tensor(key_species_indices, dtype=torch.long), persistent=False)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        if key_mode == "formation-consumption":
            self.formation_key_head = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, n_key),
            )
            self.consumption_key_head = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, n_key),
            )
        else:
            self.delta_key_head = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, n_key),
            )
        self.temperature_pressure_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.time_event_head = LatentTimeEventHead(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def encode(self, x, log_dt):
        return self.encoder(torch.cat([x, log_dt], dim=-1))

    def forward(self, x_current, log_dt, current_species=None):
        z = self.encode(x_current, log_dt)
        raw_key = None
        if self.key_mode == "direct":
            raw_key = self.delta_key_head(z)
            delta_key = raw_key
        elif self.key_mode == "signed-power":
            raw_key = self.delta_key_head(z)
            delta_key = signed_power_inverse(
                raw_key,
                alpha=self.transform_alpha,
                eps=self.transform_eps,
                scale_by_alpha=self.transform_scale_by_alpha,
            )
        else:
            formation_key = torch.nn.functional.softplus(self.formation_key_head(z))
            consumption_key = torch.nn.functional.softplus(self.consumption_key_head(z))
            delta_key = formation_key - consumption_key
            if self.key_scale_alpha != 0.0:
                if current_species is None:
                    current_species_for_scale = x_current[..., 2:]
                else:
                    current_species_for_scale = current_species
                key_species = current_species_for_scale[..., self.key_species_indices].to(delta_key.dtype)
                scale = (torch.abs(key_species) + self.transform_eps) ** self.key_scale_alpha
                delta_key = delta_key * scale
        delta_y = delta_key.to(torch.float64) @ self.completion_matrix.T
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
        next_y = current_species + delta_y
        next_state = torch.cat([next_tp, next_y], dim=-1)
        time_event = self.time_event_head(z)
        return {
            "next_state": next_state,
            "delta_key": delta_key,
            "raw_key": raw_key,
            "delta_y": delta_y,
            "z": z,
            "log_dt": time_event["log_dt"],
            "equilibrium_logit": time_event["equilibrium_logit"],
        }


class StoichiometricFluxModel(torch.nn.Module):
    """Predict reaction extents and update species through W * S @ xi."""

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        latent_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        stoich = torch.as_tensor(stoichiometric_mass_matrix, dtype=torch.float64)
        if stoich.ndim != 2:
            raise ValueError("stoichiometric_mass_matrix must be 2D")
        n_species, n_reactions = stoich.shape
        if input_dim != 2 + n_species:
            raise ValueError(f"input_dim {input_dim} is inconsistent with {n_species} species")

        self.input_dim = input_dim
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.register_buffer("stoichiometric_mass_matrix", stoich)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.flux_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_reactions),
        )
        self.temperature_pressure_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.time_event_head = LatentTimeEventHead(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def encode(self, x, log_dt):
        return self.encoder(torch.cat([x, log_dt], dim=-1))

    def forward(self, x_current, log_dt, current_species=None):
        z = self.encode(x_current, log_dt)
        reaction_extent = self.flux_head(z)
        delta_y = reaction_extent.to(torch.float64) @ self.stoichiometric_mass_matrix.T
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
        next_y = current_species + delta_y
        next_state = torch.cat([next_tp, next_y], dim=-1)
        time_event = self.time_event_head(z)
        return {
            "next_state": next_state,
            "reaction_extent": reaction_extent,
            "delta_y": delta_y,
            "z": z,
            "log_dt": time_event["log_dt"],
            "equilibrium_logit": time_event["equilibrium_logit"],
        }


def build_autoencoder_latent_gru(*, model_config, n_species: int, device):
    params = model_config.params
    input_dim = int(params.get("input_dim", 2 + n_species))
    latent_dim = int(params.get("latent_dim", 16))
    hidden_dim = int(params.get("hidden_dim", 128))
    num_layers = int(params.get("num_layers", 1))
    condition_dim = int(params.get("condition_dim", 0))
    return AutoencoderLatentGRU(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        condition_dim=condition_dim,
    ).to(device)
