"""Fit a diagonal distance without evaluation features or evaluation labels."""
import numpy as np
from scipy.stats import rankdata


def calibration_mask(library_ids, excluded_training_ids):
    if len(np.unique(library_ids)) != len(library_ids):
        raise ValueError("Library IDs must be unique")
    if not np.isin(excluded_training_ids, library_ids).all():
        raise ValueError("Training query missing from library")
    return ~np.isin(library_ids, excluded_training_ids)


def rank_weights(features, targets, uniform_fraction=0.1):
    """Return species-by-feature weights and signed marginal rank correlations.

    The caller passes training calibration arrays only. Constant columns have
    zero correlation. A constant target receives the unchanged global metric.
    """
    x, y = np.asarray(features), np.asarray(targets)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not 0 < uniform_fraction <= 1):
        raise ValueError("Invalid calibration arrays or uniform weight fraction")
    xr, yr = rankdata(x, axis=0), rankdata(y, axis=0)
    xr -= xr.mean(axis=0)
    yr -= yr.mean(axis=0)
    denominator = np.linalg.norm(yr, axis=0)[:, None]*np.linalg.norm(xr, axis=0)[None, :]
    correlation = np.divide(yr.T @ xr, denominator, out=np.zeros((y.shape[1],x.shape[1])),
                            where=denominator > 0)
    importance = np.square(correlation)
    total = importance.sum(axis=1, keepdims=True)
    normalized = np.divide(importance*x.shape[1],total,out=np.ones_like(importance),where=total>0)
    weights = uniform_fraction+(1-uniform_fraction)*normalized
    if not np.isfinite(weights).all() or np.any(weights < uniform_fraction):
        raise ValueError("Invalid fitted metric")
    return weights, correlation


def scale_features(features, weights):
    if (weights.ndim != 1 or len(weights) != features.shape[1]
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError("Require a finite positive diagonal metric")
    return features.astype(np.float64)*np.sqrt(weights)[None, :]
