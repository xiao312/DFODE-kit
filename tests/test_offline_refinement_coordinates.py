import numpy as np
import pytest

from benchmarks.offline_accuracy.refinement.coordinates import decode, encode
from benchmarks.offline_accuracy.refinement.plan import configuration, variants
from benchmarks.offline_accuracy.refinement.arithmetic import residual_round_trip


@pytest.mark.parametrize("target", ["budget-log", "budget-asinh", "residual"])
def test_signed_round_trip(target):
    values = np.r_[0., np.logspace(-46, -2, 80), -np.logspace(-46, -2, 80)]
    encoded = encode(np.ones_like(values), values, target, 1e-12)
    actual, correction = decode(np.ones_like(values), encoded, target, 1e-12)
    np.testing.assert_allclose(actual, values, rtol=3e-14, atol=0)
    assert not correction.any()
    assert actual[0] == 0


@pytest.mark.parametrize("target", ["budget-log", "budget-asinh"])
def test_tolerance_coordinate_derivative(target):
    values = np.array([-1e-8, -1e-14, 0., 1e-14, 1e-8])
    step = np.maximum(abs(values), 1e-14)*1e-5
    derivative = (encode(None, values+step, target, 1)-encode(None, values-step, target, 1))/(2*step)
    expected = 1/(1e-15+.1*abs(values)) if target == "budget-log" else 1/np.hypot(1e-15, .1*values)
    np.testing.assert_allclose(derivative, expected, rtol=6e-6)


@pytest.mark.parametrize("target", ["budget-log", "budget-asinh"])
def test_inverse_overflow_is_not_clipped(target):
    with pytest.raises(ValueError, match="overflow"):
        decode(None, [1e5], target, 1)
    with pytest.raises(ValueError, match="finite"):
        encode(None, [np.nan], target, 1)


def test_matrix_has_all_controls():
    rows = variants()
    assert len(rows) == 10
    assert len({r["name"] for r in rows}) == 10
    assert sum(r["residual"] for r in rows) == 1
    assert sum(r["physical_weight"] > 0 for r in rows) == 2
    for row in rows:
        config = configuration(row["name"], 20261011)
        assert config["training_count"] == 10000
        assert config["atol"] == 1e-15
        assert config["rtol"] == .1
    with pytest.raises(ValueError):
        configuration(rows[0]["name"], 12)
    with pytest.raises(ValueError):
        configuration("selected-after-results", 20261011)


def test_residual_arithmetic_does_not_claim_lost_digits_are_recovered():
    result = residual_round_trip([[1e-3, 0.]], [[1e-20, 0.]], ["A", "AR"])
    assert result["nonzero_reconstructed_as_zero"] == 1
    assert result["maximum_absolute_error"] == 1e-20
    primary = next(row for row in result["acceptance"]["tolerances"] if row["atol"] == 1e-15 and row["rtol"] == .1)
    assert primary["state_pass_count"] == 1  # Passing the floor is not relative precision.
