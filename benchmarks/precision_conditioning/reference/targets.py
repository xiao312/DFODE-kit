"""Three target coordinates with FP64 reconstruction and train-only scale fitting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def finite(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite")
    return values


def positive_scale(scale):
    scale = finite(scale)
    if np.any(scale <= 0):
        raise ValueError("Scales must be positive")
    return scale


def encode(delta, kind, scale=1.0, power=0.1, dtype=np.float64):
    delta = finite(delta)
    if kind == "budget-linear":
        result = delta / positive_scale(scale)
    elif kind == "signed-power":
        if not np.isfinite(power) or not 0 < power <= 1:
            raise ValueError("Power must be in (0, 1]")
        result = np.sign(delta) * np.abs(delta)**power / power
    elif kind == "scaled-asinh":
        scale = positive_scale(scale)
        delta, scale = np.broadcast_arrays(delta, scale)
        result = np.zeros_like(delta)
        active = delta != 0
        log_ratio = np.log(np.abs(delta[active])) - np.log(scale[active])
        # Avoid overflow in delta / scale. The asymptotic error above 350 is
        # below FP64 resolution; small arguments use NumPy's stable arcsinh.
        encoded = np.empty_like(log_ratio)
        large = log_ratio > 350
        encoded[large] = log_ratio[large] + np.log(2)
        encoded[~large] = np.arcsinh(delta[active][~large] / scale[active][~large]) * np.sign(delta[active][~large])
        result[active] = np.sign(delta[active]) * encoded
    else:
        raise ValueError(f"Unknown target coordinate: {kind}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError("Encoded coordinate is not finite in the requested dtype")
    return result


def decode(encoded, kind, scale=1.0, power=0.1):
    encoded = finite(encoded)
    if kind == "budget-linear":
        result = encoded * positive_scale(scale)
    elif kind == "signed-power":
        if not np.isfinite(power) or not 0 < power <= 1:
            raise ValueError("Power must be in (0, 1]")
        with np.errstate(over="ignore"):
            result = np.sign(encoded) * (np.abs(encoded) * power)**(1 / power)
    elif kind == "scaled-asinh":
        encoded, scale = np.broadcast_arrays(encoded, positive_scale(scale))
        result = np.zeros_like(encoded)
        large = np.abs(encoded) > 20
        result[~large] = np.sinh(encoded[~large]) * scale[~large]
        magnitude = np.abs(encoded[large])
        log_value = np.log(scale[large]) + magnitude - np.log(2) + np.log1p(-np.exp(-2*magnitude))
        with np.errstate(over="ignore"):
            result[large] = np.sign(encoded[large]) * np.exp(log_value)
    else:
        raise ValueError(f"Unknown target coordinate: {kind}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Decoded increment is not finite")
    return result


def training_rows(values, split):
    values = finite(values)
    split = np.asarray(split)
    if values.ndim != 2 or split.shape != (len(values),):
        raise ValueError("Expected a sample-by-feature matrix and one split per sample")
    if not np.all(np.isin(split, ["train", "validation", "test"])):
        raise ValueError("Unknown split")
    selected = values[split == "train"]
    if not len(selected):
        raise ValueError("No training samples")
    return selected


def fit_asinh_scale(delta, split, quantile=0.9, floor=1e-32):
    if not 0 < quantile <= 1 or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Invalid scale fit options")
    selected = training_rows(delta, split)
    return np.maximum(np.quantile(np.abs(selected), quantile, axis=0), floor)


@dataclass
class TrainOnlyNormalizer:
    """Scale-only outputs preserve the origin; centered inputs are optional."""
    offset: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, split, center=False):
        selected = training_rows(values, split)
        offset = np.mean(selected, axis=0) if center else np.zeros(selected.shape[1])
        magnitude = np.max(np.abs(selected - offset), axis=0)
        magnitude = np.where(magnitude == 0, 1.0, magnitude)
        scale = magnitude * np.sqrt(np.mean(((selected - offset) / magnitude)**2, axis=0))
        scale = np.where(scale == 0, 1.0, scale)
        return cls(offset, scale)

    def transform(self, values, dtype=np.float64):
        return finite((finite(values) - self.offset) / self.scale).astype(dtype)

    def inverse(self, values):
        return finite(values) * self.scale + self.offset


def physical_errors(predicted, reference, initial_y, relative_mask):
    predicted, reference, initial_y = map(finite, (predicted, reference, initial_y))
    if predicted.shape != reference.shape or initial_y.shape != reference.shape:
        raise ValueError("Physical arrays must have equal shapes")
    mask = np.asarray(relative_mask, dtype=bool)
    if mask.shape != reference.shape:
        raise ValueError("Relative mask must match reference shape")
    mask = mask & (reference != 0)
    absolute = np.abs(predicted - reference)
    weighted = absolute / (1e-12 + 1e-6 * np.abs(initial_y))
    relative = absolute[mask] / np.abs(reference[mask])
    return {"budget_rms": float(np.sqrt(np.mean(weighted**2))),
            "budget_p99": float(np.quantile(weighted, 0.99)),
            "budget_max": float(weighted.max()), "budget_exceedance": float(np.mean(weighted > 1)),
            "relative_count": int(mask.sum()), "relative_excluded_fraction": float(1-mask.mean()),
            "relative_p99": float(np.quantile(relative, 0.99)) if relative.size else None}


def validate_group_splits(trajectory_ids, split):
    groups = {}
    for trajectory_id, label in zip(trajectory_ids, split, strict=True):
        if label not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if trajectory_id in groups and groups[trajectory_id] != label:
            raise ValueError("A parent trajectory occurs in multiple splits")
        groups[trajectory_id] = label
    if set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("Three independent trajectory splits are required before training")
