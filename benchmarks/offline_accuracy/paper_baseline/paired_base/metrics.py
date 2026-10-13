"""Explicit common-row alignment and reference-free equal-weight means."""
import numpy as np


def align_indices(source_ids, requested):
    source_ids = np.asarray(source_ids)
    requested = np.asarray(requested)
    if source_ids.ndim != 1 or requested.ndim != 1:
        raise ValueError("Expected one-dimensional row IDs")
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(requested)) != len(requested):
        raise ValueError("Duplicate source IDs")
    order = np.argsort(source_ids)
    positions = np.searchsorted(source_ids[order], requested)
    if np.any(positions >= len(source_ids)) or not np.array_equal(source_ids[order[positions]], requested):
        raise ValueError("Missing source IDs")
    return order[positions]


def means(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Invalid paired predictions")
    return {"physical-mean": .5 * first + .5 * second,
            "asinh-mean": 1e-14 * np.sinh(.5 * np.arcsinh(first / 1e-14) + .5 * np.arcsinh(second / 1e-14))}


def agreement(first, second, truth, mask):
    a, b, d = first[mask], second[mask], truth[mask]
    if not len(d):
        return dict(components=0, correlation=None)
    allowance = 1e-15 + .1 * abs(d)
    pa, pb = abs(a-d) <= allowance, abs(b-d) <= allowance
    e1 = np.arcsinh(a / 1e-14) - np.arcsinh(d / 1e-14)
    e2 = np.arcsinh(b / 1e-14) - np.arcsinh(d / 1e-14)
    correlation = None if e1.std() == 0 or e2.std() == 0 else float(np.corrcoef(e1, e2)[0, 1])
    return dict(components=len(d), correlation=correlation,
                both_pass=int((pa & pb).sum()), first_only=int((pa & ~pb).sum()),
                second_only=int((~pa & pb).sum()), both_fail=int((~pa & ~pb).sum()),
                opposite_error=int((((a < d) & (b > d)) | ((a > d) & (b < d))).sum()),
                both_fail_opposite=int((~pa & ~pb & (np.sign(a-d) != np.sign(b-d))).sum()),
                oracle_union_pass=int((pa | pb).sum()))


def bounded_mean(anchor, other, fraction=.1):
    """Declared prediction-space trust region; this is NOT the scoring allowance.

    Stay on the segment from anchor to its physical midpoint with other. The
    input-only cap bounds movement by fraction*(1e-15+.1*abs(anchor)).
    With an exact base error E, E_new <= E + fraction*(1+.1*E).
    Thus E <= (1-fraction)/(1+.1*fraction) is sufficient for preservation.
    No labels or unknown acceptance flags enter this inference operation.
    """
    anchor, other = np.asarray(anchor, dtype=np.float64), np.asarray(other, dtype=np.float64)
    if anchor.shape != other.shape or not np.isfinite(anchor).all() or not np.isfinite(other).all():
        raise ValueError("Invalid predictions")
    if not 0 < fraction < 1:
        raise ValueError("Require a finite strict fractional bound")
    change = .5 * other - .5 * anchor
    cap = fraction * (1e-15 + .1 * np.abs(anchor))
    return anchor + np.minimum(np.maximum(change, -cap), cap)


def shrink_only(anchor, bounded_prediction):
    """Declared sign/magnitude-preserving restriction of an existing update.

    Project onto the physical segment between anchor and zero. This cannot
    increase |prediction|, hence cannot introduce a new severe-overprediction
    event at any fixed magnitude threshold. It cannot fix a wrong base sign.
    This is an experimental inference restriction, not a change to scoring.
    """
    anchor, bounded_prediction = np.asarray(anchor), np.asarray(bounded_prediction)
    if anchor.shape != bounded_prediction.shape or not np.isfinite(anchor).all() or not np.isfinite(bounded_prediction).all():
        raise ValueError("Invalid predictions")
    return np.minimum(np.maximum(bounded_prediction, np.minimum(anchor, 0.)), np.maximum(anchor, 0.))


def strict_shrink(anchor, bounded_prediction, fraction=.1):
    """Keep between anchor and (1-fraction)*anchor, preserving nonzero sign.

    Combined with bounded_mean this reduces, never enlarges, that movement.
    This cannot fix base wrong signs; it also cannot create new wrong signs.
    """
    if not 0 < fraction < 1: raise ValueError("Invalid fraction")
    anchor, bounded_prediction = np.asarray(anchor), np.asarray(bounded_prediction)
    if anchor.shape != bounded_prediction.shape or not np.isfinite(anchor).all() or not np.isfinite(bounded_prediction).all():
        raise ValueError("Invalid predictions")
    inner = (1-fraction)*anchor
    return np.minimum(np.maximum(bounded_prediction, np.minimum(anchor, inner)), np.maximum(anchor, inner))
