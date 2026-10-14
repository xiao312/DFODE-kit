"""Separate residual target/reconstruction rounding from learned correction error."""
import numpy as np

from benchmarks.offline_accuracy.metrics import summarize


def residual_round_trip(base, reference, species):
    base, reference = np.asarray(base, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if base.shape != reference.shape or reference.ndim != 2 or not np.isfinite(base).all() or not np.isfinite(reference).all():
        raise ValueError("Require matching finite state-by-species matrices")
    residual = reference-base
    reconstructed = base+residual
    active = np.array([name != "AR" for name in species])
    observed, truth = reconstructed[:, active], reference[:, active]
    nonzero = truth != 0
    return dict(scope="FP64 arithmetic only; exact residual before network approximation",
                nonzero_components=int(nonzero.sum()),
                nonzero_reconstructed_as_zero=int(((observed == 0) & nonzero).sum()),
                changed_components=int((observed != truth).sum()),
                maximum_absolute_error=float(np.max(abs(observed-truth))),
                acceptance=summarize(reconstructed, reference, species))
