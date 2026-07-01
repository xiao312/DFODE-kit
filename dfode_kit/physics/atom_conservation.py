from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompletionMatrix:
    matrix: np.ndarray
    key_species_indices: tuple[int, ...]
    dependent_species_indices: tuple[int, ...]
    element_rank: int


@dataclass(frozen=True)
class ReactionStoichiometry:
    net: np.ndarray
    reactants: np.ndarray
    products: np.ndarray
    mass_fraction_matrix: np.ndarray
    reversible: np.ndarray


def atom_molecule_matrix(gas) -> np.ndarray:
    """Return element-by-species atom counts."""

    matrix = np.zeros((gas.n_elements, gas.n_species), dtype=np.float64)
    for elem_idx in range(gas.n_elements):
        for species_idx in range(gas.n_species):
            matrix[elem_idx, species_idx] = gas.n_atoms(species_idx, elem_idx)
    return matrix


def mass_fraction_element_matrix(gas) -> np.ndarray:
    """Return element-balance matrix for species mass-fraction updates.

    For mass fractions Y, elemental mole inventory per unit mixture mass is:

        E = Y @ (N / W).T

    Therefore a mass-fraction update dY conserves atoms exactly when
    (N / W) @ dY = 0.
    """

    atoms = atom_molecule_matrix(gas)
    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    return atoms / molecular_weights[None, :]


def _independent_columns(matrix: np.ndarray, tol: float = 1e-12) -> list[int]:
    independent: list[int] = []
    basis = np.empty((matrix.shape[0], 0), dtype=np.float64)
    rank = 0
    for col_idx in range(matrix.shape[1]):
        candidate = np.column_stack([basis, matrix[:, col_idx]])
        candidate_rank = np.linalg.matrix_rank(candidate, tol=tol)
        if candidate_rank > rank:
            independent.append(col_idx)
            basis = candidate
            rank = candidate_rank
        if rank == matrix.shape[0]:
            break
    return independent


def completion_matrix(atom_matrix: np.ndarray, tol: float = 1e-12) -> CompletionMatrix:
    """Build a key-species completion matrix C with atom_matrix @ C == 0.

    Source updates are represented as s_dot = C @ s_dot_key. The key species
    rows form an identity matrix; dependent species rows complete atom balance.
    """

    atom_matrix = np.asarray(atom_matrix, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(atom_matrix, tol=tol))
    n_species = atom_matrix.shape[1]
    n_key = n_species - rank
    if n_key <= 0:
        raise ValueError("No atom-conserving degrees of freedom available")

    dependent = _independent_columns(atom_matrix, tol=tol)
    if len(dependent) != rank:
        raise ValueError(f"Expected {rank} dependent species, found {len(dependent)}")

    dependent_set = set(dependent)
    key = [idx for idx in range(n_species) if idx not in dependent_set]

    a_dep = atom_matrix[:, dependent]
    a_key = atom_matrix[:, key]
    dependent_block = -np.linalg.solve(a_dep, a_key)

    matrix = np.zeros((n_species, n_key), dtype=np.float64)
    for local_idx, species_idx in enumerate(key):
        matrix[species_idx, local_idx] = 1.0
    for local_dep_idx, species_idx in enumerate(dependent):
        matrix[species_idx, :] = dependent_block[local_dep_idx, :]

    residual = atom_matrix @ matrix
    if not np.allclose(residual, 0.0, atol=1e-8, rtol=1e-8):
        raise ValueError("Completion matrix failed atom-conservation check")

    return CompletionMatrix(
        matrix=matrix,
        key_species_indices=tuple(key),
        dependent_species_indices=tuple(dependent),
        element_rank=rank,
    )


def completion_matrix_for_mass_fractions(gas, tol: float = 1e-12) -> CompletionMatrix:
    return completion_matrix(mass_fraction_element_matrix(gas), tol=tol)


def stoichiometric_mass_fraction_matrix(gas) -> np.ndarray:
    """Return W * S for reaction extents per unit mixture mass.

    S is product stoichiometry minus reactant stoichiometry with shape
    (n_species, n_reactions). If xi has units of kmol reaction extent per kg
    mixture, then delta_Y = (W * S) @ xi.
    """

    stoich = np.asarray(gas.product_stoich_coeffs - gas.reactant_stoich_coeffs, dtype=np.float64)
    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    return molecular_weights[:, None] * stoich


def reaction_stoichiometry(gas) -> ReactionStoichiometry:
    """Return species-by-reaction stoichiometric matrices and reversibility."""

    reactants = np.asarray(gas.reactant_stoich_coeffs, dtype=np.float64)
    products = np.asarray(gas.product_stoich_coeffs, dtype=np.float64)
    net = products - reactants
    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    reversible = np.asarray([bool(gas.reaction(idx).reversible) for idx in range(gas.n_reactions)], dtype=bool)
    return ReactionStoichiometry(
        net=net,
        reactants=reactants,
        products=products,
        mass_fraction_matrix=molecular_weights[:, None] * net,
        reversible=reversible,
    )


def reaction_affinity_over_rt(stoich_net: np.ndarray, chemical_potentials_over_rt: np.ndarray) -> np.ndarray:
    """Return dimensionless reaction affinities A / RT.

    For species chemical potentials mu_i and net stoichiometry S_ij
    (products minus reactants), reaction affinity is:

        A_j = -sum_i S_ij * mu_i

    Passing mu_i / RT returns A_j / RT. Positive affinity corresponds to the
    thermodynamically favored forward direction under the supplied state.
    """

    stoich_net = np.asarray(stoich_net, dtype=np.float64)
    mu_over_rt = np.asarray(chemical_potentials_over_rt, dtype=np.float64)
    return -(mu_over_rt @ stoich_net)


def element_totals_from_mole_amounts(gas, mole_amounts: np.ndarray) -> np.ndarray:
    atoms = atom_molecule_matrix(gas)
    return np.asarray(mole_amounts, dtype=np.float64) @ atoms.T


def element_totals_from_mass_fractions(gas, mass_fractions: np.ndarray) -> np.ndarray:
    y = np.asarray(mass_fractions, dtype=np.float64)
    mole_amounts_per_mass = y / np.asarray(gas.molecular_weights, dtype=np.float64)
    return element_totals_from_mole_amounts(gas, mole_amounts_per_mass)


def mass_fraction_sum_error(mass_fractions: np.ndarray) -> np.ndarray:
    y = np.asarray(mass_fractions, dtype=np.float64)
    return np.abs(np.sum(y, axis=-1) - 1.0)


def negative_mass_fraction_rate(mass_fractions: np.ndarray, threshold: float = 0.0) -> float:
    y = np.asarray(mass_fractions, dtype=np.float64)
    return float(np.mean(y < threshold))
