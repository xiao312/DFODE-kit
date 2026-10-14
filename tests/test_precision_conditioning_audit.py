import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "precision_audit", Path(__file__).parents[1] / "benchmarks/precision_conditioning/audit.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("coordinate", ["budget-linear", "budget-asinh", "signed-power"])
def test_round_trip_preserves_tiny_signed_increments(coordinate):
    increments = np.array([-1e-32, 0.0, 1e-32, 1e-4])
    decoded = audit.round_trip(increments, np.full(4, 1e-12), coordinate, np.float64)
    np.testing.assert_allclose(decoded, increments, rtol=5e-14, atol=0)


def test_known_increment_can_disappear_from_endpoint():
    report = audit.run_audit()
    assert report["state_addition_lost_nonzero"] > 0
    endpoint = report["results"][0]
    assert endpoint["nonzero_predicted_zero"] > 0
    assert endpoint["relative_error_max"] >= 1.0


@pytest.mark.parametrize("atol,rtol", [(0, 1e-6), (float("nan"), 0), (1e-12, -1)])
def test_invalid_budget_rejected(atol, rtol):
    with pytest.raises(ValueError):
        audit.run_audit(atol, rtol)
