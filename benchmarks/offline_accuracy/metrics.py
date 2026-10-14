"""Physical increment tolerance counts, with explicit reference qualification."""
from __future__ import annotations

import numpy as np

ABSOLUTE_FLOORS = (1e-12, 1e-15, 1e-18)
RELATIVE_TOLERANCES = (1., .1, .01, .001)
MAGNITUDE_EDGES = (0., 1e-30, 1e-20, 1e-15, 1e-12, 1e-9, 1e-6, np.inf)


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def tolerance_counts(prediction, reference, atol, rtol, uncertainty=None):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if (prediction.shape != reference.shape or reference.ndim != 2 or not reference.size
            or not np.isfinite(reference).all() or not np.isfinite(prediction).all()):
        raise ValueError("Require matching nonempty finite state-by-species matrices")
    if not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError("Require a positive absolute floor and nonnegative relative tolerance")
    budget = atol + rtol * np.abs(reference)
    error = np.abs(prediction - reference)
    nominal = error <= budget
    result = {"atol": atol, "rtol": rtol, "components": reference.size, "states": len(reference),
              "component_pass_count": int(nominal.sum()), "state_pass_count": int(nominal.all(axis=1).sum()),
              "component_pass_fraction": float(nominal.mean()), "state_pass_fraction": float(nominal.all(axis=1).mean()),
              "normalized_error_p99": float(np.quantile(error / budget, .99)),
              "normalized_error_max": float(np.max(error / budget))}
    if uncertainty is not None:
        uncertainty = np.asarray(uncertainty)
        if uncertainty.shape != reference.shape or np.any(uncertainty < 0) or np.isinf(uncertainty).any():
            raise ValueError("Uncertainty must match references; unknown values must be NaN")
        known = np.isfinite(uncertainty) & (uncertainty <= .1 * budget)
        # Empirical margin, not a certified bound on an unknown exact solution.
        passed = known & (error + uncertainty <= budget)
        known_states = known.all(axis=1)
        result.update(qualified_components=int(known.sum()), unknown_components=int((~known).sum()),
                      qualified_pass_count=int(passed.sum()),
                      qualified_pass_fraction=fraction(passed.sum(), known.sum()),
                      qualified_states=int(known_states.sum()), unknown_states=int((~known_states).sum()),
                      qualified_state_pass_count=int(passed.all(axis=1).sum()),
                      qualified_state_pass_fraction=fraction(passed.all(axis=1).sum(), known_states.sum()))
    return result


def summarize(prediction, reference, species, uncertainty=None):
    active = np.array([name != "AR" for name in species])
    if not active.any() or len(species) != np.asarray(reference).shape[1]:
        raise ValueError("Require matching species names and a non-argon output")
    prediction, reference = np.asarray(prediction)[:, active], np.asarray(reference)[:, active]
    uncertainty = None if uncertainty is None else np.asarray(uncertainty)[:, active]
    rows = [tolerance_counts(prediction, reference, absolute, relative, uncertainty)
            for absolute in ABSOLUTE_FLOORS for relative in RELATIVE_TOLERANCES]
    small = np.abs(reference) < 1e-15
    small_pass = small & (np.abs(prediction) < 1e-15)
    per_species = []
    for index, name in enumerate(np.asarray(species)[active]):
        u = None if uncertainty is None else uncertainty[:, index:index+1]
        per_species.append({"species": str(name), **tolerance_counts(prediction[:, index:index+1],
            reference[:, index:index+1], 1e-15, .1, u)})
    bins = []
    for lower, upper in zip(MAGNITUDE_EDGES[:-1], MAGNITUDE_EDGES[1:]):
        mask = (np.abs(reference) >= lower) & (np.abs(reference) < upper)
        values = np.abs(prediction - reference)[mask]
        passes = values <= 1e-15 + .1 * np.abs(reference[mask])
        bins.append({"lower": lower, "upper": upper if np.isfinite(upper) else None,
                     "components": int(mask.sum()), "pass_count": int(passes.sum()),
                     "pass_fraction": fraction(passes.sum(), mask.sum()),
                     "absolute_error_p99": float(np.quantile(values, .99)) if values.size else None})
    return {"tolerances": rows, "per_species": per_species, "magnitude_bins": bins,
            "sspi": {"threshold": 1e-15, "small_components": int(small.sum()),
                     "small_predictions": int(small_pass.sum()), "value": fraction(small_pass.sum(), small.sum())}}
