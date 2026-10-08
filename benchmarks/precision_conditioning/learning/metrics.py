"""Physical-space metrics; the same denominators apply to every coordinate."""
import numpy as np


def error_summary(predicted, reference, weights, relative_mask):
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Non-finite predictions cannot be scored")
    absolute = np.abs(predicted - reference)
    budget = absolute / weights
    mask = relative_mask & (reference != 0)
    relative = absolute[mask] / np.abs(reference[mask])
    return {"count": int(reference.size), "budget_p50": float(np.median(budget)),
            "budget_p99": float(np.quantile(budget, .99)), "budget_max": float(budget.max()),
            "budget_rms": float(np.sqrt(np.mean(budget**2))), "budget_exceedance": float(np.mean(budget > 1)),
            "relative_count": int(mask.sum()), "relative_p99": float(np.quantile(relative, .99)) if relative.size else None}


def evaluate(predicted, data, selected):
    reference, weights = data["delta"][selected], data["weights"][selected]
    initial, mask = data["initial"][selected], data["relative_mask"][selected]
    result = error_summary(predicted[:, 1:], reference[:, 1:], weights[:, 1:], mask[:, 1:])
    result["temperature_budget_p99"] = float(np.quantile(np.abs(predicted[:, 0] - reference[:, 0]) / weights[:, 0], .99))
    result["mass_delta_sum_abs_max"] = float(np.abs(predicted[:, 1:].sum(axis=1)).max())
    result["minimum_predicted_Y"] = float((initial[:, 1:] + predicted[:, 1:]).min())
    result["negative_endpoint_fraction"] = float(np.mean(initial[:, 1:] + predicted[:, 1:] < 0))
    result["species"] = [error_summary(predicted[:, i:i+1], reference[:, i:i+1], weights[:, i:i+1], mask[:, i:i+1])
                         for i in range(1, reference.shape[1])]
    result["magnitude_bins"] = []
    magnitudes = np.abs(reference[:, 1:])
    for lower, upper in [(0., 1e-32), (1e-32, 1e-24), (1e-24, 1e-16), (1e-16, 1e-8), (1e-8, np.inf)]:
        chosen = (magnitudes >= lower) & (magnitudes < upper) & (magnitudes != 0)
        item = {"lower": lower, "upper": upper if np.isfinite(upper) else None, "count": int(chosen.sum())}
        if chosen.any():
            item.update(error_summary(predicted[:, 1:][chosen], reference[:, 1:][chosen], weights[:, 1:][chosen], mask[:, 1:][chosen]))
        result["magnitude_bins"].append(item)
    return result
