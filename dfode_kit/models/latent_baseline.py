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


class HardConservationLayer(torch.nn.Module):
    """Complete key-species deltas into mass/atom-conserving species updates."""

    def __init__(self, completion_matrix):
        super().__init__()
        completion = torch.as_tensor(completion_matrix, dtype=torch.float64)
        if completion.ndim != 2:
            raise ValueError("completion_matrix must be 2D")
        self.n_species, self.n_key = completion.shape
        self.register_buffer("completion_matrix", completion)

    def forward(self, delta_key, current_species):
        delta_y = delta_key.to(torch.float64) @ self.completion_matrix.T
        next_y = current_species.to(torch.float64) + delta_y
        return next_y, delta_y


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
        self.hard_conservation = HardConservationLayer(completion)
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
        if current_species is None:
            current_species = x_current[..., 2:]
        next_y, delta_y = self.hard_conservation(delta_key, current_species)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
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
        flux_mode: str = "signed-power",
        transform_alpha: float = 0.1,
        transform_scale_by_alpha: bool = True,
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
        self.flux_mode = flux_mode
        self.transform_alpha = transform_alpha
        self.transform_scale_by_alpha = transform_scale_by_alpha
        if flux_mode not in {"direct", "signed-power"}:
            raise ValueError(f"Unsupported flux_mode: {flux_mode}")
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
        raw_reaction_extent = self.flux_head(z)
        if self.flux_mode == "direct":
            reaction_extent = raw_reaction_extent
        else:
            reaction_extent = signed_power_inverse(
                raw_reaction_extent,
                alpha=self.transform_alpha,
                scale_by_alpha=self.transform_scale_by_alpha,
            )
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
            "raw_reaction_extent": raw_reaction_extent,
            "reaction_extent": reaction_extent,
            "delta_y": delta_y,
            "z": z,
            "log_dt": time_event["log_dt"],
            "equilibrium_logit": time_event["equilibrium_logit"],
        }


class StoichiometricIntervalModel(torch.nn.Module):
    """Predict integrated reaction extents for a requested time interval.

    Unlike StoichiometricFluxModel, this CFD-oriented baseline treats log_dt as
    an input condition only. It does not predict a next time step or event time.
    """

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        flux_mode: str = "signed-power",
        transform_alpha: float = 0.1,
        transform_scale_by_alpha: bool = True,
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
        self.flux_mode = flux_mode
        self.transform_alpha = transform_alpha
        self.transform_scale_by_alpha = transform_scale_by_alpha
        if flux_mode not in {"direct", "signed-power"}:
            raise ValueError(f"Unsupported flux_mode: {flux_mode}")
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

    def encode(self, x, log_dt):
        return self.encoder(torch.cat([x, log_dt], dim=-1))

    def forward(self, x_current, log_dt, current_species=None):
        z = self.encode(x_current, log_dt)
        raw_reaction_extent = self.flux_head(z)
        if self.flux_mode == "direct":
            reaction_extent = raw_reaction_extent
        else:
            reaction_extent = signed_power_inverse(
                raw_reaction_extent,
                alpha=self.transform_alpha,
                scale_by_alpha=self.transform_scale_by_alpha,
            )
        delta_y = reaction_extent.to(torch.float64) @ self.stoichiometric_mass_matrix.T
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
        next_y = current_species + delta_y
        next_state = torch.cat([next_tp, next_y], dim=-1)
        return {
            "next_state": next_state,
            "raw_reaction_extent": raw_reaction_extent,
            "reaction_extent": reaction_extent,
            "delta_y": delta_y,
            "z": z,
        }


class ThermoStoichiometricIntervalModel(torch.nn.Module):
    """Affinity-gated stoichiometric interval model.

    The network predicts nonnegative reaction mobilities. Thermodynamic
    affinity controls reaction direction for reversible reactions, irreversible
    reactions are forward-only, and the stoichiometric matrix maps integrated
    extents to atom-conserving mass-fraction updates.
    """

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        reaction_reversible,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        extent_scale: float = 1e-3,
        force_scale: float = 1.0,
        availability_threshold: float = -60.0,
        availability_slope: float = 1.0,
    ):
        super().__init__()
        stoich = torch.as_tensor(stoichiometric_mass_matrix, dtype=torch.float64)
        reversible = torch.as_tensor(reaction_reversible, dtype=torch.bool)
        if stoich.ndim != 2:
            raise ValueError("stoichiometric_mass_matrix must be 2D")
        n_species, n_reactions = stoich.shape
        if reversible.numel() != n_reactions:
            raise ValueError("reaction_reversible length must match n_reactions")
        if input_dim != 2 + n_species:
            raise ValueError(f"input_dim {input_dim} is inconsistent with {n_species} species")

        self.input_dim = input_dim
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.extent_scale = float(extent_scale)
        self.force_scale = float(force_scale)
        self.availability_threshold = float(availability_threshold)
        self.availability_slope = float(availability_slope)
        self.register_buffer("stoichiometric_mass_matrix", stoich)
        self.register_buffer("reaction_reversible", reversible)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.mobility_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_reactions),
        )
        self.temperature_pressure_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def encode(self, x, log_dt):
        return self.encoder(torch.cat([x, log_dt], dim=-1))

    def forward(self, x_current, log_dt, affinity_hat, log_reactant_activity, current_species=None):
        z = self.encode(x_current, log_dt)
        raw_mobility = self.mobility_head(z)
        mobility = torch.nn.functional.softplus(raw_mobility).to(torch.float64)
        affinity = affinity_hat.to(torch.float64)
        log_activity = log_reactant_activity.to(torch.float64)

        reversible_force = torch.tanh(affinity / max(self.force_scale, 1e-30))
        irreversible_force = torch.ones_like(reversible_force)
        force = torch.where(self.reaction_reversible[None, :], reversible_force, irreversible_force)
        availability_gate = torch.sigmoid(self.availability_slope * (log_activity - self.availability_threshold))
        reaction_extent = self.extent_scale * mobility * availability_gate * force

        delta_y = reaction_extent @ self.stoichiometric_mass_matrix.T
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
        next_y = current_species + delta_y
        next_state = torch.cat([next_tp, next_y], dim=-1)
        return {
            "next_state": next_state,
            "raw_mobility": raw_mobility,
            "mobility": mobility,
            "reaction_extent": reaction_extent,
            "force": force,
            "availability_gate": availability_gate,
            "delta_y": delta_y,
            "z": z,
        }


class LatentSubstepStoichiometricIntervalModel(torch.nn.Module):
    """Shared-weight latent substep integrator with stoichiometric updates."""

    STEP_OPTIONS = (1, 2, 4, 8)

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        substeps: int = 2,
        flux_mode: str = "signed-power",
        transform_alpha: float = 0.1,
        transform_scale_by_alpha: bool = True,
        thermo_input_dim: int = 0,
        thermo_embedding_dim: int = 32,
        reaction_extent_raw_clip: float = 0.0,
    ):
        super().__init__()
        stoich = torch.as_tensor(stoichiometric_mass_matrix, dtype=torch.float64)
        if stoich.ndim != 2:
            raise ValueError("stoichiometric_mass_matrix must be 2D")
        n_species, n_reactions = stoich.shape
        if input_dim != 2 + n_species:
            raise ValueError(f"input_dim {input_dim} is inconsistent with {n_species} species")
        if substeps < 1:
            raise ValueError("substeps must be positive")
        if flux_mode not in {"direct", "signed-power"}:
            raise ValueError(f"Unsupported flux_mode: {flux_mode}")

        self.input_dim = input_dim
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.substeps = int(substeps)
        self.flux_mode = flux_mode
        self.transform_alpha = transform_alpha
        self.transform_scale_by_alpha = transform_scale_by_alpha
        self.thermo_input_dim = int(thermo_input_dim)
        self.thermo_embedding_dim = int(thermo_embedding_dim) if thermo_input_dim > 0 else 0
        self.reaction_extent_raw_clip = float(reaction_extent_raw_clip)
        self.register_buffer("stoichiometric_mass_matrix", stoich)

        condition_dim = 1 + self.thermo_embedding_dim
        if self.thermo_input_dim > 0:
            self.thermo_encoder = torch.nn.Sequential(
                torch.nn.Linear(self.thermo_input_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, self.thermo_embedding_dim),
                torch.nn.GELU(),
            )
        else:
            self.thermo_encoder = None

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.step_net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.step_gate = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
            torch.nn.Sigmoid(),
        )
        self.flux_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_reactions),
        )
        self.temperature_pressure_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )
        self.error_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Softplus(),
        )
        self.step_policy_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + condition_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, len(self.STEP_OPTIONS)),
        )

    def _thermo_embedding(self, thermo_features):
        if self.thermo_encoder is None:
            return None
        if thermo_features is None:
            raise ValueError("thermo_features are required for this model")
        return self.thermo_encoder(thermo_features)

    def _condition(self, log_dt_value, thermo_embedding):
        if thermo_embedding is None:
            return log_dt_value
        return torch.cat([log_dt_value, thermo_embedding], dim=-1)

    def _reaction_extent_from_raw(self, raw_reaction_extent):
        if self.reaction_extent_raw_clip > 0.0:
            raw_reaction_extent = torch.clamp(
                raw_reaction_extent,
                min=-self.reaction_extent_raw_clip,
                max=self.reaction_extent_raw_clip,
            )
        if self.flux_mode == "direct":
            return raw_reaction_extent
        return signed_power_inverse(
            raw_reaction_extent,
            alpha=self.transform_alpha,
            scale_by_alpha=self.transform_scale_by_alpha,
        )

    def forward(
        self,
        x_current,
        log_dt,
        current_species=None,
        *,
        log_dt_step=None,
        substeps: int | None = None,
        thermo_features=None,
        return_substep_states: bool = False,
    ):
        n_substeps = int(substeps or self.substeps)
        if n_substeps < 1:
            raise ValueError("substeps must be positive")
        if current_species is None:
            current_species = x_current[..., 2:]
        current_species = current_species.to(torch.float64)
        if log_dt_step is None:
            log_dt_step = log_dt

        thermo_embedding = self._thermo_embedding(thermo_features)
        total_condition = self._condition(log_dt, thermo_embedding)
        step_condition = self._condition(log_dt_step, thermo_embedding)

        z = self.encoder(torch.cat([x_current, total_condition], dim=-1))
        step_policy_logits = self.step_policy_head(torch.cat([z, total_condition], dim=-1))
        y = current_species
        delta_y_total = torch.zeros_like(y, dtype=torch.float64)
        reaction_extents = []
        raw_extents = []
        substep_y = []

        for _idx in range(n_substeps):
            step_input = torch.cat([z, step_condition], dim=-1)
            dz = torch.tanh(self.step_net(step_input))
            gate = self.step_gate(step_input)
            z = z + gate * dz
            flux_input = torch.cat([z, step_condition], dim=-1)
            raw_reaction_extent = self.flux_head(flux_input)
            reaction_extent = self._reaction_extent_from_raw(raw_reaction_extent)
            delta_y = reaction_extent.to(torch.float64) @ self.stoichiometric_mass_matrix.T
            y = y + delta_y
            delta_y_total = delta_y_total + delta_y
            raw_extents.append(raw_reaction_extent)
            reaction_extents.append(reaction_extent)
            if return_substep_states:
                substep_y.append(y)

        tp_input = torch.cat([z, step_condition], dim=-1)
        next_tp = self.temperature_pressure_head(tp_input).to(torch.float64)
        error_estimate = self.error_head(tp_input).to(torch.float64)
        next_state = torch.cat([next_tp, y], dim=-1)
        output = {
            "next_state": next_state,
            "raw_reaction_extent": torch.stack(raw_extents, dim=1),
            "reaction_extent": torch.stack(reaction_extents, dim=1),
            "delta_y": delta_y_total,
            "z": z,
            "substeps": n_substeps,
            "error_estimate": error_estimate,
            "step_policy_logits": step_policy_logits,
        }
        if return_substep_states:
            output["substep_y"] = torch.stack(substep_y, dim=1)
        return output

    def predict_substeps(self, x_current, log_dt, *, thermo_features=None):
        thermo_embedding = self._thermo_embedding(thermo_features)
        total_condition = self._condition(log_dt, thermo_embedding)
        z = self.encoder(torch.cat([x_current, total_condition], dim=-1))
        logits = self.step_policy_head(torch.cat([z, total_condition], dim=-1))
        indices = torch.argmax(logits, dim=-1)
        options = torch.as_tensor(self.STEP_OPTIONS, dtype=torch.long, device=x_current.device)
        return options[indices], logits


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


class ThermodynamicProgressSubstepModel(torch.nn.Module):
    """Thermodynamic reaction-progress substep model.

    The model predicts nonnegative reaction mobilities and combines them with
    local dimensionless reaction affinities A/RT to produce reaction extents.
    Species updates are applied through W*S, preserving elemental conservation
    when the stoichiometric matrix is atom-balanced. A scalar finite-step
    limiter scales each proposed update to keep mass fractions nonnegative.

    This class is additive and does not alter existing model constructors, so
    older checkpoints keep their original class compatibility.
    """

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        reversible=None,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        n_substeps: int = 4,
        progress_mode: str = "sinh",
        affinity_clip: float = 20.0,
        positivity_safety: float = 0.999,
        mobility_floor: float = 0.0,
        mobility_scale: float = 1.0,
        latent_update_mode: str = "residual",
        latent_step_scale: float = 1.0,
        proximal_correction_steps: int = 0,
        proximal_correction_lr: float = 0.1,
        proximal_free_energy_weight: float = 0.0,
        proximal_y_floor: float = 1e-300,
        include_affinity_features: bool = True,
    ):
        super().__init__()
        stoich_mass = torch.as_tensor(stoichiometric_mass_matrix, dtype=torch.float64)
        if stoich_mass.ndim != 2:
            raise ValueError("stoichiometric_mass_matrix must be 2D")
        n_species, n_reactions = stoich_mass.shape
        if input_dim != 2 + n_species:
            raise ValueError(f"input_dim {input_dim} is inconsistent with {n_species} species")
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        if progress_mode not in {"sinh", "linear-affinity", "monotone-sinh", "monotone-tanh"}:
            raise ValueError(f"Unsupported progress_mode: {progress_mode}")
        if latent_update_mode not in {"residual", "bounded"}:
            raise ValueError(f"Unsupported latent_update_mode: {latent_update_mode}")
        if not 0.0 < positivity_safety <= 1.0:
            raise ValueError("positivity_safety must be in (0, 1]")

        self.input_dim = input_dim
        self.n_species = n_species
        self.n_reactions = n_reactions
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_substeps = n_substeps
        self.progress_mode = progress_mode
        self.affinity_clip = affinity_clip
        self.positivity_safety = positivity_safety
        self.mobility_floor = mobility_floor
        self.mobility_scale = mobility_scale
        self.latent_update_mode = latent_update_mode
        self.latent_step_scale = latent_step_scale
        if proximal_correction_steps < 0:
            raise ValueError("proximal_correction_steps must be >= 0")
        if proximal_correction_lr < 0.0:
            raise ValueError("proximal_correction_lr must be >= 0")
        if proximal_free_energy_weight < 0.0:
            raise ValueError("proximal_free_energy_weight must be >= 0")
        if proximal_y_floor <= 0.0:
            raise ValueError("proximal_y_floor must be > 0")
        self.proximal_correction_steps = proximal_correction_steps
        self.proximal_correction_lr = proximal_correction_lr
        self.proximal_free_energy_weight = proximal_free_energy_weight
        self.proximal_y_floor = proximal_y_floor
        self.include_affinity_features = include_affinity_features

        self.register_buffer("stoichiometric_mass_matrix", stoich_mass)
        if reversible is None:
            reversible_tensor = torch.ones(n_reactions, dtype=torch.bool)
        else:
            reversible_tensor = torch.as_tensor(reversible, dtype=torch.bool)
            if reversible_tensor.numel() != n_reactions:
                raise ValueError("reversible must have one entry per reaction")
        self.register_buffer("reversible", reversible_tensor.reshape(n_reactions), persistent=False)

        affinity_feature_dim = n_reactions if include_affinity_features else 0
        encoder_input_dim = input_dim + 1 + affinity_feature_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        step_input_dim = latent_dim + 1 + affinity_feature_dim
        self.step_net = torch.nn.Sequential(
            torch.nn.Linear(step_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, latent_dim),
        )
        self.mobility_head = torch.nn.Sequential(
            torch.nn.Linear(step_input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, n_reactions),
        )
        self.temperature_pressure_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def _prepare_affinity(self, affinity_over_rt, reference):
        if affinity_over_rt is None:
            raise ValueError("affinity_over_rt is required for ThermodynamicProgressSubstepModel")
        affinity = affinity_over_rt.to(device=reference.device, dtype=reference.dtype)
        if affinity.shape[-1] != self.n_reactions:
            raise ValueError(f"affinity_over_rt last dim must be {self.n_reactions}")
        return torch.clamp(affinity, min=-self.affinity_clip, max=self.affinity_clip)

    def _condition(self, z_or_x, log_dt, affinity):
        parts = [z_or_x, log_dt]
        if self.include_affinity_features:
            parts.append(affinity)
        return torch.cat(parts, dim=-1)

    def _reaction_force(self, affinity):
        reversible = self.reversible.to(device=affinity.device)
        if self.progress_mode in {"sinh", "monotone-sinh"}:
            reversible_force = 2.0 * torch.sinh(0.5 * affinity)
        elif self.progress_mode == "monotone-tanh":
            reversible_force = torch.tanh(affinity)
        else:
            reversible_force = affinity
        irreversible_force = reversible_force if self.progress_mode.startswith("monotone-") else torch.ones_like(affinity)
        return torch.where(reversible.view(*([1] * (affinity.ndim - 1)), -1), reversible_force, irreversible_force)

    def _advance_latent(self, z, step_condition):
        dz = self.step_net(step_condition)
        if self.latent_update_mode == "bounded":
            return torch.tanh(z + self.latent_step_scale * torch.tanh(dz))
        return z + dz

    def _limit_delta_y(self, current_species, delta_y):
        current = current_species.to(torch.float64)
        delta = delta_y.to(torch.float64)
        negative_update = delta < 0.0
        ratio = torch.where(
            negative_update,
            -current / torch.clamp(delta, max=-1e-300),
            torch.full_like(delta, float("inf")),
        )
        max_alpha = torch.amin(ratio, dim=-1, keepdim=True)
        alpha = torch.minimum(torch.ones_like(max_alpha), self.positivity_safety * max_alpha)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        limited_delta = alpha * delta
        return current + limited_delta, limited_delta, alpha

    def _project_extent_to_positive(self, current_species, extent):
        delta_y = extent.to(torch.float64) @ self.stoichiometric_mass_matrix.T
        y, limited_delta_y, alpha = self._limit_delta_y(current_species, delta_y)
        limited_extent = alpha * extent.to(torch.float64)
        return y, limited_delta_y, limited_extent, alpha

    def _ideal_mixture_free_energy(self, species):
        y = torch.clamp(species.to(torch.float64), min=self.proximal_y_floor)
        return torch.sum(y * torch.log(y), dim=-1, keepdim=True)

    def _ideal_mixture_mu(self, species):
        y = torch.clamp(species.to(torch.float64), min=self.proximal_y_floor)
        return torch.log(y) + 1.0

    def _proximal_correct_extent(self, current_species, extent_guess):
        if self.proximal_correction_steps <= 0 or self.proximal_free_energy_weight <= 0.0:
            y, limited_delta_y, limited_extent, alpha = self._project_extent_to_positive(current_species, extent_guess)
            return y, limited_delta_y, limited_extent, alpha

        current = current_species.to(torch.float64)
        guess = extent_guess.to(torch.float64)
        extent = guess
        alpha = torch.ones((current.shape[0], 1), dtype=torch.float64, device=current.device)
        for _ in range(self.proximal_correction_steps):
            y, _limited_delta_y, extent, alpha = self._project_extent_to_positive(current, extent)
            mu = self._ideal_mixture_mu(y)
            grad_free_energy = mu @ self.stoichiometric_mass_matrix
            grad = (extent - guess) + self.proximal_free_energy_weight * grad_free_energy
            extent = extent - self.proximal_correction_lr * grad

        y, limited_delta_y, limited_extent, alpha = self._project_extent_to_positive(current, extent)
        return y, limited_delta_y, limited_extent, alpha

    def encode(self, x_current, log_dt, affinity_over_rt):
        affinity = self._prepare_affinity(affinity_over_rt, x_current)
        return self.encoder(self._condition(x_current, log_dt, affinity))

    def forward(
        self,
        x_current,
        log_dt,
        affinity_over_rt,
        log_reactant_activity=None,
        current_species=None,
        log_dt_step=None,
        substeps=None,
    ):
        if current_species is None:
            current_species = x_current[..., 2:]
        affinity = self._prepare_affinity(affinity_over_rt, x_current)
        z = self.encoder(self._condition(x_current, log_dt, affinity))
        y = current_species.to(torch.float64)
        initial_ideal_free_energy = self._ideal_mixture_free_energy(y)
        n_substeps = int(substeps) if substeps is not None else int(self.n_substeps)
        if n_substeps < 1:
            raise ValueError("substeps must be >= 1")
        if log_dt_step is None:
            h_log_dt = log_dt - torch.log(torch.as_tensor(float(n_substeps), device=log_dt.device, dtype=log_dt.dtype))
        else:
            h_log_dt = log_dt_step

        total_delta_y = torch.zeros_like(y, dtype=torch.float64)
        total_affinity_work = torch.zeros((y.shape[0], 1), dtype=torch.float64, device=y.device)
        extent_steps = []
        limiter_steps = []
        affinity_work_steps = []
        for _ in range(n_substeps):
            step_condition = self._condition(z, h_log_dt, affinity)
            z = self._advance_latent(z, step_condition)
            mobility = torch.nn.functional.softplus(self.mobility_head(step_condition)) + self.mobility_floor
            force = self._reaction_force(affinity)
            extent = (self.mobility_scale * mobility * force / float(n_substeps)).to(torch.float64)
            y, limited_delta_y, limited_extent, alpha = self._proximal_correct_extent(y, extent)
            affinity_work = torch.sum(limited_extent * affinity.to(torch.float64), dim=-1, keepdim=True)
            total_delta_y = total_delta_y + limited_delta_y
            total_affinity_work = total_affinity_work + affinity_work
            extent_steps.append(limited_extent)
            limiter_steps.append(alpha)
            affinity_work_steps.append(affinity_work)

        next_tp = self.temperature_pressure_head(z).to(torch.float64)
        next_state = torch.cat([next_tp, y], dim=-1)
        ideal_free_energy_delta = self._ideal_mixture_free_energy(y) - initial_ideal_free_energy
        return {
            "next_state": next_state,
            "next_y": y,
            "next_tp": next_tp,
            "delta_y": total_delta_y,
            "reaction_extent": torch.stack(extent_steps, dim=-2),
            "positivity_alpha": torch.stack(limiter_steps, dim=-2),
            "affinity_work": total_affinity_work,
            "affinity_work_steps": torch.stack(affinity_work_steps, dim=-2),
            "local_free_energy_proxy_delta": -total_affinity_work,
            "ideal_mixture_free_energy_delta": ideal_free_energy_delta,
            "z": z,
            "affinity_over_rt": affinity,
        }


class ThermoProgressSubstepStoichiometricIntervalModel(ThermodynamicProgressSubstepModel):
    """Compatibility name used by stoich-interval training/evaluation code."""

    def __init__(
        self,
        input_dim: int,
        stoichiometric_mass_matrix,
        reaction_reversible=None,
        reversible=None,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        substeps: int = 4,
        n_substeps: int | None = None,
        force_clip: float = 20.0,
        affinity_clip: float | None = None,
        mobility_scale: float = 1.0,
        positivity_safety_factor: float = 0.999,
        positivity_safety: float | None = None,
        progress_mode: str = "sinh",
        mobility_floor: float = 0.0,
        latent_update_mode: str = "residual",
        latent_step_scale: float = 1.0,
        proximal_correction_steps: int = 0,
        proximal_correction_lr: float = 0.1,
        proximal_free_energy_weight: float = 0.0,
        proximal_y_floor: float = 1e-300,
        include_affinity_features: bool = True,
    ):
        if reversible is None:
            reversible = reaction_reversible
        if n_substeps is None:
            n_substeps = substeps
        if affinity_clip is None:
            affinity_clip = force_clip
        if positivity_safety is None:
            positivity_safety = positivity_safety_factor
        if positivity_safety <= 0.0:
            positivity_safety = 0.999
        super().__init__(
            input_dim=input_dim,
            stoichiometric_mass_matrix=stoichiometric_mass_matrix,
            reversible=reversible,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_substeps=n_substeps,
            progress_mode=progress_mode,
            affinity_clip=affinity_clip,
            positivity_safety=positivity_safety,
            mobility_floor=mobility_floor,
            mobility_scale=mobility_scale,
            latent_update_mode=latent_update_mode,
            latent_step_scale=latent_step_scale,
            proximal_correction_steps=proximal_correction_steps,
            proximal_correction_lr=proximal_correction_lr,
            proximal_free_energy_weight=proximal_free_energy_weight,
            proximal_y_floor=proximal_y_floor,
            include_affinity_features=include_affinity_features,
        )
