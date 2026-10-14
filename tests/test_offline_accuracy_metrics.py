import numpy as np
import pytest
import copy

from benchmarks.offline_accuracy.metrics import summarize, tolerance_counts
from benchmarks.offline_accuracy.verify_evaluation import check_summary


@pytest.mark.parametrize("field", ["per_species", "magnitude_bins"])
def test_independent_check_rejects_missing_diagnostics(field):
    reference = np.array([[1e-12, 1e-6]])
    result = copy.deepcopy(summarize(reference, reference, ["A", "B"]))
    result[field].pop()
    with pytest.raises(ValueError):
        check_summary(reference, reference, result, ["A", "B"])


def test_primary_tolerance_and_whole_state_are_distinct():
    reference = np.array([[1e-12, 1e-6], [1e-12, 1e-6]])
    prediction = reference + [[1e-13, 1e-7], [1.1e-13, 0]]
    row = tolerance_counts(prediction, reference, 1e-15, .1)
    assert row["component_pass_count"] == 3 and row["state_pass_count"] == 1
    assert row["component_pass_fraction"] == .75 and row["state_pass_fraction"] == .5


def test_zero_has_perfect_sspi_but_fails_significant_increments():
    reference = np.array([[1e-32, 1e-6, 0.], [-1e-20, -1e-7, 0.]])
    result = summarize(np.zeros_like(reference), reference, ["A", "B", "AR"])
    assert result["sspi"]["value"] == 1
    primary = next(row for row in result["tolerances"] if row["atol"] == 1e-15 and row["rtol"] == .1)
    assert primary["component_pass_fraction"] == .5 and primary["state_pass_fraction"] == 0
    assert any(row["pass_fraction"] is None for row in result["magnitude_bins"])


def test_unknown_reference_is_not_counted_as_qualified_pass():
    ref = np.array([[1e-12, 1e-6], [1e-12, 1e-6]])
    uncertainty = np.array([[np.nan, 1e-12], [1e-14, 1e-4]])
    result = tolerance_counts(ref, ref, 1e-15, .1, uncertainty)
    assert result["component_pass_count"] == 4
    assert result["qualified_components"] == 2 and result["qualified_pass_count"] == 2
    assert result["qualified_states"] == 0 and result["qualified_state_pass_fraction"] is None


def test_tolerance_monotonicity_and_sspi_empty_group():
    ref = np.array([[1e-6, 2e-6]])
    predicted = ref * 1.05
    result = summarize(predicted, ref, ["A", "B"])
    assert result["sspi"]["value"] is None
    for absolute in (1e-12, 1e-15, 1e-18):
        counts = [r["component_pass_count"] for r in result["tolerances"] if r["atol"] == absolute]
        assert counts == sorted(counts, reverse=True)
    with pytest.raises(ValueError):
        tolerance_counts(predicted * np.nan, ref, 1e-15, .1)


def test_independent_verifier_checks_counts_and_reference_unknowns():
    ref = np.array([[1e-32, 1e-6, 0.], [-1e-20, -1e-7, 0.]])
    prediction = ref * 1.02
    uncertainty = np.array([[np.nan, 1e-14, 0.], [1e-25, 1e-15, 0.]])
    result = summarize(prediction, ref, ["A", "B", "AR"], uncertainty)
    check_summary(prediction, ref, result, ["A", "B", "AR"], uncertainty)
    result["tolerances"][0]["component_pass_count"] -= 1
    with pytest.raises(ValueError):
        check_summary(prediction, ref, result, ["A", "B", "AR"], uncertainty)
