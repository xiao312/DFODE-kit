import numpy as np
import pytest

from dfode_kit.evaluation.metrics import a_index, mae, r2_score, rmse, sspi, summarize_predictions


def test_basic_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])

    assert mae(y_true, y_pred) == pytest.approx(1.0 / 3.0)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.5 / 3.0))
    assert r2_score(y_true, y_pred) == pytest.approx(0.75)


def test_sspi_counts_predictions_that_remain_small():
    y_true = np.array([1e-16, 5e-16, 1e-9, 2e-15])
    y_pred = np.array([2e-16, 1e-12, 1e-9, 1e-16])

    assert sspi(y_true, y_pred, threshold=1e-15) == pytest.approx(0.5)


def test_a_index_uses_relative_error():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([1.1, 2.5, 4.2])

    assert a_index(y_true, y_pred, tolerance=0.20) == pytest.approx(2 / 3)


def test_summary_includes_overall_and_per_species_sspi():
    y_true = np.array([[1e-16, 1.0], [2e-16, 2.0]])
    y_pred = np.array([[1e-16, 1.1], [1e-9, 1.9]])

    summary = summarize_predictions(y_true, y_pred, species_names=["A", "B"], small_thresholds=(1e-15,))

    assert summary["overall"]["sspi_1e-15"] == pytest.approx(0.5)
    assert summary["per_species"][0]["species"] == "A"
    assert summary["per_species"][0]["sspi_1e-15"] == pytest.approx(0.5)
