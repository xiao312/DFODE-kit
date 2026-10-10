import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

PATH = Path(__file__).parents[1] / "benchmarks/precision_conditioning/reference/targets.py"
spec = importlib.util.spec_from_file_location("precision_targets", PATH)
targets = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = targets
spec.loader.exec_module(targets)


@pytest.mark.parametrize("kind", ["budget-linear", "signed-power", "scaled-asinh"])
@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 5e-14), (np.float32, 2e-5)])
def test_signed_round_trip(kind, dtype, tolerance):
    delta = np.r_[-np.logspace(-32, -2, 50), 0, np.logspace(-32, -2, 50)]
    encoded = targets.encode(delta, kind, scale=1e-12, dtype=dtype)
    decoded = targets.decode(encoded, kind, scale=1e-12)
    np.testing.assert_allclose(decoded, delta, rtol=tolerance, atol=0)
    assert decoded[50] == 0


def test_asinh_handles_extreme_ratio_without_intermediate_overflow():
    delta = np.array([-1e200, 0, 1e200])
    encoded = targets.encode(delta, "scaled-asinh", scale=1e-200)
    np.testing.assert_allclose(targets.decode(encoded, "scaled-asinh", scale=1e-200), delta, rtol=2e-13)


def test_scales_do_not_use_held_out_data():
    values = np.array([[1e-10, 0], [2e-10, 0], [1e10, 1e10]])
    split = ["train", "train", "test"]
    normalizer = targets.TrainOnlyNormalizer.fit(values, split)
    changed = values.copy()
    changed[-1] *= 1000
    other = targets.TrainOnlyNormalizer.fit(changed, split)
    np.testing.assert_array_equal(normalizer.scale, other.scale)
    np.testing.assert_array_equal(targets.fit_asinh_scale(values, split), targets.fit_asinh_scale(changed, split))
    assert np.all(normalizer.transform(np.zeros((1, 2))) == 0)


def test_group_leakage_and_missing_splits_rejected():
    with pytest.raises(ValueError, match="multiple"):
        targets.validate_group_splits(["a", "a", "b"], ["train", "test", "validation"])
    with pytest.raises(ValueError, match="Three"):
        targets.validate_group_splits(["a", "b"], ["train", "test"])
    targets.validate_group_splits(["a", "b", "c"], ["train", "validation", "test"])


def test_common_physical_evaluation_masks_unresolved_relative_error():
    result = targets.physical_errors(np.array([[0., 0.]]), np.array([[1e-32, 1e-4]]),
                                     np.array([[0., .1]]), np.array([[False, True]]))
    assert result["relative_count"] == 1
    assert result["relative_p99"] == 1
    assert result["relative_excluded_fraction"] == .5


@pytest.mark.parametrize("kind", ["budget-linear", "scaled-asinh"])
def test_invalid_scales_rejected(kind):
    with pytest.raises(ValueError):
        targets.encode([1.0], kind, scale=0)
