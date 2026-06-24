from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


DEFAULT_SMALL_THRESHOLDS = (1e-15, 1e-12)
DEFAULT_A_TOLERANCES = (0.01, 0.05, 0.10, 0.20, 1.00)


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true has {true.shape}, y_pred has {pred.shape}")
    return true, pred


def mae(y_true, y_pred, axis=None) -> np.ndarray:
    true, pred = _validate_pair(y_true, y_pred)
    return np.mean(np.abs(pred - true), axis=axis)


def rmse(y_true, y_pred, axis=None) -> np.ndarray:
    true, pred = _validate_pair(y_true, y_pred)
    return np.sqrt(np.mean((pred - true) ** 2, axis=axis))


def r2_score(y_true, y_pred, axis=None) -> np.ndarray:
    true, pred = _validate_pair(y_true, y_pred)
    residual = np.sum((true - pred) ** 2, axis=axis)
    centered = true - np.mean(true, axis=axis, keepdims=True)
    total = np.sum(centered ** 2, axis=axis)
    return np.divide(
        total - residual,
        total,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=total > 0,
    )


def sspi(y_true, y_pred, threshold: float = 1e-15, axis=None) -> np.ndarray:
    """Small-Scale Prediction Index.

    SSPI is the fraction of true small-scale targets whose predictions are also
    small-scale targets under the same absolute threshold.
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive")

    true, pred = _validate_pair(y_true, y_pred)
    true_small = np.abs(true) < threshold
    pred_small = np.abs(pred) < threshold
    numerator = np.sum(true_small & pred_small, axis=axis)
    denominator = np.sum(true_small, axis=axis)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def sspi_counts(y_true, y_pred, threshold: float = 1e-15, axis=None) -> dict[str, np.ndarray]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    true, pred = _validate_pair(y_true, y_pred)
    true_small = np.abs(true) < threshold
    pred_small = np.abs(pred) < threshold
    return {
        "true_small": np.sum(true_small, axis=axis),
        "pred_small": np.sum(pred_small, axis=axis),
        "both_small": np.sum(true_small & pred_small, axis=axis),
    }


def a_index(y_true, y_pred, tolerance: float = 0.20, eps: float = 1e-300, axis=None) -> np.ndarray:
    """Fraction of predictions with absolute relative error below tolerance."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    true, pred = _validate_pair(y_true, y_pred)
    rel_error = np.abs(pred - true) / np.maximum(np.abs(true), eps)
    return np.mean(rel_error <= tolerance, axis=axis)


def per_species_summary(
    y_true,
    y_pred,
    species_names: Sequence[str] | None = None,
    *,
    small_thresholds: Iterable[float] = DEFAULT_SMALL_THRESHOLDS,
    a_tolerances: Iterable[float] = DEFAULT_A_TOLERANCES,
) -> list[dict[str, float | str]]:
    true, pred = _validate_pair(y_true, y_pred)
    if true.ndim != 2:
        raise ValueError(f"Expected 2D arrays shaped (n_samples, n_species), got {true.shape}")

    n_species = true.shape[1]
    names = list(species_names) if species_names is not None else [f"species_{i}" for i in range(n_species)]
    if len(names) != n_species:
        raise ValueError(f"Expected {n_species} species names, got {len(names)}")

    rows = []
    for idx, name in enumerate(names):
        row: dict[str, float | str] = {
            "species": name,
            "mae": float(mae(true[:, idx], pred[:, idx])),
            "rmse": float(rmse(true[:, idx], pred[:, idx])),
            "r2": float(r2_score(true[:, idx], pred[:, idx])),
        }
        for threshold in small_thresholds:
            row[f"sspi_{threshold:g}"] = float(sspi(true[:, idx], pred[:, idx], threshold=threshold))
            counts = sspi_counts(true[:, idx], pred[:, idx], threshold=threshold)
            row[f"sspi_{threshold:g}_true_small"] = int(counts["true_small"])
            row[f"sspi_{threshold:g}_pred_small"] = int(counts["pred_small"])
            row[f"sspi_{threshold:g}_both_small"] = int(counts["both_small"])
            small_mask = np.abs(true[:, idx]) < threshold
            if np.any(small_mask):
                abs_pred_small_targets = np.abs(pred[:, idx][small_mask])
                row[f"sspi_{threshold:g}_median_abs_pred_on_true_small"] = float(
                    np.median(abs_pred_small_targets)
                )
                row[f"sspi_{threshold:g}_min_abs_pred_on_true_small"] = float(np.min(abs_pred_small_targets))
            else:
                row[f"sspi_{threshold:g}_median_abs_pred_on_true_small"] = float("nan")
                row[f"sspi_{threshold:g}_min_abs_pred_on_true_small"] = float("nan")
        for tolerance in a_tolerances:
            row[f"a{int(tolerance * 100):g}"] = float(a_index(true[:, idx], pred[:, idx], tolerance=tolerance))
        rows.append(row)

    return rows


def summarize_predictions(
    y_true,
    y_pred,
    species_names: Sequence[str] | None = None,
    *,
    small_thresholds: Iterable[float] = DEFAULT_SMALL_THRESHOLDS,
    a_tolerances: Iterable[float] = DEFAULT_A_TOLERANCES,
) -> dict:
    true, pred = _validate_pair(y_true, y_pred)
    flat_true = true.reshape(-1)
    flat_pred = pred.reshape(-1)

    summary = {
        "overall": {
            "mae": float(mae(flat_true, flat_pred)),
            "rmse": float(rmse(flat_true, flat_pred)),
            "r2": float(r2_score(flat_true, flat_pred)),
        },
        "per_species": per_species_summary(
            true.reshape(-1, true.shape[-1]),
            pred.reshape(-1, pred.shape[-1]),
            species_names=species_names,
            small_thresholds=small_thresholds,
            a_tolerances=a_tolerances,
        ),
    }

    for threshold in small_thresholds:
        summary["overall"][f"sspi_{threshold:g}"] = float(sspi(flat_true, flat_pred, threshold=threshold))
        counts = sspi_counts(flat_true, flat_pred, threshold=threshold)
        summary["overall"][f"sspi_{threshold:g}_true_small"] = int(counts["true_small"])
        summary["overall"][f"sspi_{threshold:g}_pred_small"] = int(counts["pred_small"])
        summary["overall"][f"sspi_{threshold:g}_both_small"] = int(counts["both_small"])
        small_mask = np.abs(flat_true) < threshold
        if np.any(small_mask):
            abs_pred_small_targets = np.abs(flat_pred[small_mask])
            summary["overall"][f"mae_small_{threshold:g}"] = float(mae(flat_true[small_mask], flat_pred[small_mask]))
            summary["overall"][f"rmse_small_{threshold:g}"] = float(rmse(flat_true[small_mask], flat_pred[small_mask]))
            summary["overall"][f"sspi_{threshold:g}_median_abs_pred_on_true_small"] = float(
                np.median(abs_pred_small_targets)
            )
            summary["overall"][f"sspi_{threshold:g}_min_abs_pred_on_true_small"] = float(
                np.min(abs_pred_small_targets)
            )
            summary["overall"][f"sspi_{threshold:g}_p10_abs_pred_on_true_small"] = float(
                np.quantile(abs_pred_small_targets, 0.10)
            )
        else:
            summary["overall"][f"mae_small_{threshold:g}"] = float("nan")
            summary["overall"][f"rmse_small_{threshold:g}"] = float("nan")
            summary["overall"][f"sspi_{threshold:g}_median_abs_pred_on_true_small"] = float("nan")
            summary["overall"][f"sspi_{threshold:g}_min_abs_pred_on_true_small"] = float("nan")
            summary["overall"][f"sspi_{threshold:g}_p10_abs_pred_on_true_small"] = float("nan")

    for tolerance in a_tolerances:
        summary["overall"][f"a{int(tolerance * 100):g}"] = float(a_index(flat_true, flat_pred, tolerance=tolerance))

    return summary
