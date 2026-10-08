import numpy as np
import pytest

pytest.importorskip("cantera")
pytest.importorskip("scipy")
from benchmarks.flame_conditioning.scout import summarize_record, temperature_selection


def test_temperature_selection_unique_and_spans_range():
    states = np.array([[300], [300], [500], [1000], [1500], [2000], [2000]])
    selected = temperature_selection(states, 5)
    assert len(set(selected)) == 5
    assert states[selected].min() == 300
    assert states[selected].max() == 2000
    np.testing.assert_array_equal(selected, temperature_selection(states, 5))


def test_missing_check_cannot_pass():
    assert summarize_record({"state": {"Y": [.5, .5]}, "checks": {}}) == {"status": "missing_check"}


def test_relative_mask_distinguishes_budget_fit_and_zero():
    values = [1e-12, 0.0]
    record = {"state": {"Y": [.5, .5]}, "checks": {
        name: {"delta": values} for name in
        ["tight", "tighter", "step-limited", "direct", "direct-tighter", "study-tolerance"]
    }}
    result = summarize_record(record)
    assert result["budget_fit"] == [True, True]
    assert result["relative_fit"] == [True, False]
    assert result["zero_reference_count"] == 1
    del record["checks"]["study-tolerance"]
    assert summarize_record(record)["status"] == "missing_check"
