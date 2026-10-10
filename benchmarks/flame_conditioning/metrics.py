"""Common physical-space evidence; no hidden positivity/conservation repair."""
from __future__ import annotations

import numpy as np


def summary(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p99": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Metrics require finite predictions")
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p99": float(np.quantile(values, .99)),
            "max": float(values.max())}


def physical_scores(predicted, reference, initial, corrected, species_names,
                    element_matrix, formation_enthalpies, molecular_weights, interval=1e-6):
    predicted, reference, initial = map(np.asarray, (predicted, reference, initial))
    if predicted.shape != reference.shape or predicted.shape != initial[:, 2:].shape:
        raise ValueError("Prediction and reference shapes must match state species")
    if not all(np.isfinite(array).all() for array in (predicted, reference, initial)):
        raise ValueError("Nonfinite physical evidence")
    active = np.array([name != "AR" for name in species_names])
    absolute = np.abs(predicted - reference)
    weight = 1e-12 + 1e-6 * np.abs(initial[:, 2:])
    weighted = absolute / weight
    small = np.abs(reference[:, active]) < 1e-15
    predicted_small = np.abs(predicted[:, active]) < 1e-15
    # This is an endpoint-spacing screen only. Solver convergence has been
    # independently checked on a subset, not certified component by component.
    spacing = np.maximum(np.abs(np.spacing(initial[:, 2:])),
                         np.abs(np.spacing(initial[:, 2:] + reference)))
    nominal_relative = (np.abs(reference) > 100 * spacing) & (np.abs(reference) >= 1e-15)
    nominal_relative[:, ~active] = False
    relative = absolute[nominal_relative] / np.abs(reference[nominal_relative])
    density = initial[:, 1] / (8314.46261815324 * initial[:, 0] * np.sum(initial[:, 2:] / molecular_weights, axis=1))
    heat_reference = -(reference @ formation_enthalpies) * density / interval
    # Subtract increments first: two large heat releases can hide a small error.
    heat_error = -((predicted-reference) @ formation_enthalpies) * density / interval
    result = {
        "samples": len(initial), "non_argon_species_components": int(len(initial) * active.sum()),
        "absolute_error": summary(absolute[:, active]), "budget_error": summary(weighted[:, active]),
        "budget_exceedance": float(np.mean(weighted[:, active] > 1)),
        "small_target_count": int(small.sum()),
        "small_target_stays_small_fraction": float(predicted_small[small].mean()) if small.any() else None,
        "nominal_relative_error": summary(relative),
        "nominal_relative_label": "Magnitude/endpoint-spacing screened; not individually solver-certified",
        "inverse_domain_correction_fraction": float(np.mean(np.asarray(corrected)[:, active])),
        "negative_endpoint_fraction": float(np.mean((initial[:, 2:] + predicted)[:, active] < 0)),
        "negative_endpoint_row_fraction": float(np.mean(np.any(initial[:, 2:] + predicted < 0, axis=1))),
        "mass_increment_drift": summary(np.abs(predicted.sum(axis=1))),
        "element_increment_drift": summary(np.abs(predicted @ element_matrix.T)),
        "heat_release_absolute_error_W_m3": summary(np.abs(heat_error)),
        "heat_release_reference_rms_W_m3": float(np.sqrt(np.mean(heat_reference ** 2))),
        "heat_release_error_rms_W_m3": float(np.sqrt(np.mean(heat_error ** 2))),
        "temperature_bins": [], "magnitude_bins": [], "per_species": {},
    }
    edges = [0, 305, 500, 1000, 1500, 2000, 3000]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (initial[:, 0] >= low) & (initial[:, 0] < high)
        result["temperature_bins"].append({"lower_K": low, "upper_K": high, "samples": int(mask.sum()),
                                            "budget_error": summary(weighted[mask][:, active])})
    magnitude_edges = [0, 1e-30, 1e-20, 1e-15, 1e-10, 1e-5, float("inf")]
    magnitude = np.abs(reference[:, active])
    for low, high in zip(magnitude_edges[:-1], magnitude_edges[1:], strict=True):
        mask = (magnitude >= low) & (magnitude < high)
        result["magnitude_bins"].append({"lower": low, "upper": high if np.isfinite(high) else None,
                                         "absolute_error": summary(absolute[:, active][mask])})
    for index, name in enumerate(species_names):
        if name != "AR":
            result["per_species"][name] = {"absolute_error": summary(absolute[:, index]),
                                            "budget_error": summary(weighted[:, index])}
    return result
