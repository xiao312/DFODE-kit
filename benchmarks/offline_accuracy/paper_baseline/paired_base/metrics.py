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
