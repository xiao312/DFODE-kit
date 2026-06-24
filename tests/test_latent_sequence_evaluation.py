import numpy as np

from dfode_kit.evaluation.latent_sequence import _relative_time_error


def test_relative_time_error_reports_zero_for_exact_dt():
    true_dt = np.array([1e-9, 1e-8, 1e-7])
    pred_dt = true_dt.copy()

    metrics = _relative_time_error(true_dt, pred_dt)

    assert metrics["mae_log_dt"] == 0.0
    assert metrics["mae_dt"] == 0.0
    assert metrics["median_relative_dt_error"] == 0.0
