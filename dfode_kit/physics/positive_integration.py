from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dfode_kit.physics.atom_conservation import mass_fraction_element_matrix


@dataclass(frozen=True)
class ProjectionDiagnostics:
    iterations: int
    negative_rate: float
    max_constraint_residual: float
    fallback_count: int = 0


def independent_constraint_rows(matrix: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    """Return an orthonormal row basis spanning the supplied constraints."""

    matrix = np.asarray(matrix, dtype=np.float64)
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        raise ValueError("constraint matrix is empty")
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    if rank == 0:
        raise ValueError("constraint matrix has zero numerical rank")
    return np.ascontiguousarray(vh[:rank], dtype=np.float64)


def augmented_mass_constraint_matrix(gas, tolerance: float = 1e-12) -> np.ndarray:
    """Return independent element and total-mass constraints for mass fractions."""

    atom = mass_fraction_element_matrix(gas)
    augmented = np.vstack([atom, np.ones((1, gas.n_species), dtype=np.float64)])
    return independent_constraint_rows(augmented, tolerance=tolerance)


def _constraint_residual(y: torch.Tensor, current: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
    return (y - current) @ constraint.T


def sandu_composition_projection(
    proposal: torch.Tensor,
    current: torch.Tensor,
    constraint_matrix: torch.Tensor,
    *,
    floor: float = 0.0,
    max_iterations: int = 64,
    tolerance: float = 1e-14,
) -> tuple[torch.Tensor, ProjectionDiagnostics]:
    """Euclidean projection onto positivity and conservation constraints.

    This is the composition-space quadratic program described by Sandu:

        min_y 0.5 ||y - y_hat||_2^2
        s.t.  A y = A y_n, y >= floor.

    A primal-dual active set solves only the low-dimensional equality system.
    All arithmetic in the projection is float64.
    """

    q = proposal.to(torch.float64)
    y0 = current.to(torch.float64)
    a = constraint_matrix.to(device=q.device, dtype=torch.float64)
    b = y0 @ a.T
    active = q < floor
    eye = torch.eye(a.shape[0], dtype=torch.float64, device=q.device)
    completed = 0

    for completed in range(1, max_iterations + 1):
        free = ~active
        free_f = free.to(torch.float64)
        system = torch.einsum("bi,ri,si->brs", free_f, a, a)
        fixed = torch.where(active, torch.full_like(q, floor), q)
        rhs = fixed @ a.T - b
        scale = torch.amax(torch.abs(system), dim=(1, 2), keepdim=True).clamp_min(1.0)
        lam = torch.linalg.solve(
            system + torch.finfo(torch.float64).eps * scale * eye[None],
            rhs[..., None],
        ).squeeze(-1)
        free_value = q - lam @ a
        y = torch.where(active, torch.full_like(q, floor), free_value)
        multiplier = floor - q + lam @ a
        violated = free & (y < floor)
        invalid_active = active & (multiplier < -tolerance)
        updated = (active & ~invalid_active) | violated
        if torch.equal(updated, active):
            break
        active = updated

    residual = _constraint_residual(y, y0, a)
    diagnostics = ProjectionDiagnostics(
        iterations=completed,
        negative_rate=float(torch.mean((y < floor).to(torch.float64)).detach().cpu()),
        max_constraint_residual=float(torch.max(torch.abs(residual)).detach().cpu()),
    )
    return y, diagnostics


def sandu_reaction_space_projection(
    proposed_extent: torch.Tensor,
    current: torch.Tensor,
    stoichiometric_mass_matrix: torch.Tensor,
    *,
    floor: float = 0.0,
    max_iterations: int = 64,
    tolerance: float = 1e-14,
) -> tuple[torch.Tensor, torch.Tensor, ProjectionDiagnostics]:
    """Project reaction extents onto ``current + S xi >= floor``.

    The active-set solve minimizes the change in reaction coordinates. Element
    and total-mass conservation are automatic because every update remains in
    the range of the stoichiometric matrix.
    """

    xi_hat = proposed_extent.to(torch.float64)
    y0 = current.to(torch.float64)
    s = stoichiometric_mass_matrix.to(device=xi_hat.device, dtype=torch.float64)
    interior_floor = max(float(floor), 1e-30)
    lower = interior_floor - y0
    gram = s @ s.T
    active = (xi_hat @ s.T) < lower
    eye_species = torch.eye(s.shape[0], dtype=torch.float64, device=s.device)
    xi = xi_hat
    completed = 0

    for completed in range(1, max_iterations + 1):
        active_f = active.to(torch.float64)
        system = (
            active_f[:, :, None] * gram[None] * active_f[:, None, :]
            + (1.0 - active_f)[:, :, None] * eye_species[None]
        )
        rhs = (lower - xi_hat @ s.T) * active_f
        scale = torch.amax(torch.abs(system), dim=(1, 2), keepdim=True).clamp_min(1.0)
        multipliers = torch.linalg.solve(
            system + torch.finfo(torch.float64).eps * scale * eye_species[None],
            rhs[..., None],
        ).squeeze(-1)
        xi = xi_hat + multipliers @ s
        values = xi @ s.T
        violated = values < lower
        invalid_active = active & (multipliers < -tolerance)
        updated = (active & ~invalid_active) | violated
        if torch.equal(updated, active):
            break
        active = updated

    y = y0 + xi @ s.T
    failed = torch.any(y < floor, dim=1)
    fallback_count = int(torch.sum(failed).detach().cpu())
    if fallback_count:
        u, singular_values, _vh = torch.linalg.svd(s, full_matrices=True)
        rank = int(
            torch.sum(singular_values > 1e-12 * singular_values[0]).detach().cpu()
        )
        conservation_basis = u[:, rank:].T
        fallback, _fallback_diagnostics = sandu_composition_projection(
            y[failed],
            y0[failed],
            conservation_basis,
            floor=floor,
        )
        y = y.clone()
        y[failed] = fallback
    diagnostics = ProjectionDiagnostics(
        iterations=completed,
        negative_rate=float(torch.mean((y < floor).to(torch.float64)).detach().cpu()),
        max_constraint_residual=0.0,
        fallback_count=fallback_count,
    )
    return xi, y, diagnostics


def relative_positive_projection(
    proposal: torch.Tensor,
    current: torch.Tensor,
    constraint_matrix: torch.Tensor,
    *,
    floor: float = 1e-30,
    correction_iterations: int = 4,
    tolerance: float = 1e-13,
    sandu_fallback: bool = True,
) -> tuple[torch.Tensor, ProjectionDiagnostics]:
    """Relative/positivity-weighted conservation projection.

    The metric ``W = diag(1 / q)`` penalizes relative rather than absolute
    corrections. The conservation solve has only ``rank(A)`` dimensions:

        d_corr = d - D A^T (A D A^T)^+ A d,  D = diag(q^2).
    """

    y0 = current.to(torch.float64)
    a = constraint_matrix.to(device=y0.device, dtype=torch.float64)
    q = torch.clamp(proposal.to(torch.float64), min=floor)
    eye = torch.eye(a.shape[0], dtype=torch.float64, device=q.device)
    completed = 0

    for completed in range(1, correction_iterations + 1):
        delta = q - y0
        d = q.square()
        ad = a[None] * d[:, None, :]
        system = ad @ a.T
        rhs = delta @ a.T
        scale = torch.amax(torch.abs(system), dim=(1, 2), keepdim=True).clamp_min(1.0)
        lam = torch.linalg.solve(system + tolerance * scale * eye[None], rhs[..., None]).squeeze(-1)
        corrected = delta - (lam[:, :, None] * ad).sum(dim=1)
        y = y0 + corrected
        if bool(torch.all(y >= floor - tolerance)):
            break
        q = torch.clamp(y, min=floor)

    fallback_count = 0
    if sandu_fallback:
        failed = torch.any(y < floor - tolerance, dim=1)
        fallback_count = int(torch.sum(failed).detach().cpu())
        if fallback_count:
            fallback, _ = sandu_composition_projection(
                y[failed],
                y0[failed],
                a,
                floor=floor,
                tolerance=tolerance,
            )
            y = y.clone()
            y[failed] = fallback

    residual = _constraint_residual(y, y0, a)
    diagnostics = ProjectionDiagnostics(
        iterations=completed,
        negative_rate=float(torch.mean((y < floor).to(torch.float64)).detach().cpu()),
        max_constraint_residual=float(torch.max(torch.abs(residual)).detach().cpu()),
        fallback_count=fallback_count,
    )
    return y, diagnostics
