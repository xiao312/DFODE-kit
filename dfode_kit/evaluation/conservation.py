from __future__ import annotations

import numpy as np

from dfode_kit.physics.atom_conservation import (
    atom_molecule_matrix,
    completion_matrix,
    completion_matrix_for_mass_fractions,
    element_totals_from_mass_fractions,
    mass_fraction_sum_error,
    negative_mass_fraction_rate,
)


def conservation_summary(gas, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    true_elements = element_totals_from_mass_fractions(gas, y_true)
    pred_elements = element_totals_from_mass_fractions(gas, y_pred)
    residual = pred_elements - true_elements
    abs_residual = np.abs(residual)
    completion = completion_matrix_for_mass_fractions(gas)

    return {
        "mean_mass_sum_error": float(np.mean(mass_fraction_sum_error(y_pred))),
        "max_mass_sum_error": float(np.max(mass_fraction_sum_error(y_pred))),
        "negative_mass_fraction_rate": negative_mass_fraction_rate(y_pred),
        "mean_abs_element_residual": float(np.mean(abs_residual)),
        "max_abs_element_residual": float(np.max(abs_residual)),
        "mean_l2_element_residual": float(np.mean(np.linalg.norm(residual, axis=-1))),
        "completion_matrix": {
            "n_species": int(gas.n_species),
            "n_elements": int(gas.n_elements),
            "element_rank": int(completion.element_rank),
            "basis": "mass_fraction_element_matrix",
            "n_key_species": int(len(completion.key_species_indices)),
            "key_species": [gas.species_names[idx] for idx in completion.key_species_indices],
            "dependent_species": [gas.species_names[idx] for idx in completion.dependent_species_indices],
        },
    }
