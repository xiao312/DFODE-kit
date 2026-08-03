from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
        availability_floor: float = 1e-30,
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
        self.register_buffer("reaction_reversible", torch.tensor(reversible, dtype=torch.float64))
        self.extent_scale = float(extent_scale)
        self.availability_floor = float(availability_floor)
        self.n_reactions = reactants.shape[1]
        self.encoder = _backbone(input_dim, hidden_dim, latent_dim)
        self.tp_head = nn.Linear(latent_dim, 2)
        self.demand_head = nn.Linear(latent_dim, 2 * self.n_reactions)
        nn.init.constant_(self.demand_head.bias, -8.0)

    def forward(self, current_state, log_dt, *, current_species):
        hidden = self.encoder(torch.cat([current_state, log_dt], dim=-1))
        next_tp = self.tp_head(hidden).to(torch.float64)
        demand = F.softplus(self.demand_head(hidden).to(torch.float64)) * self.extent_scale
        requested = demand @ self.consumption.T
        availability = torch.clamp(
            current_species.to(torch.float64) / requested.clamp_min(self.availability_floor),
            min=0.0,
            max=1.0,
        )
        process_availability = torch.where(
            self.consumption.T[None] > 0.0,
            availability[:, None, :],
            torch.ones((), dtype=torch.float64, device=availability.device),
        ).amin(dim=-1)
        process_availability = process_availability * (1.0 - 1e-12)
        process_extent = demand * process_availability
        delta_y = process_extent @ self.process_stoich.T
        next_y = current_species.to(torch.float64) + delta_y
        forward_extent = process_extent[:, : self.n_reactions]
        reverse_extent = process_extent[:, self.n_reactions :]
        return {
            "next_state": torch.cat([next_tp, next_y], dim=-1),
            "next_species": next_y,
            "delta_species": delta_y,
            "reaction_extent": forward_extent - reverse_extent,
            "forward_extent": forward_extent,
            "reverse_extent": reverse_extent,
            "minimum_process_availability": process_availability.amin(dim=1, keepdim=True),
        }


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
