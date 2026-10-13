import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.paired_base.metrics import align_indices, means, agreement


def test_alignment_reorders_and_rejects_missing_or_duplicate():
    np.testing.assert_array_equal(align_indices([4, 2, 9], [9, 4]), [2, 0])
    for source, query in (([1, 1], [1]), ([1, 2], [1, 1]), ([1, 2], [3]), ([2, 4], [1])):
        with pytest.raises(ValueError):
            align_indices(source, query)


def test_fixed_means_sign_symmetry_and_identical_inputs():
    value = np.array([[1e-12, -1e-3, 0.]])
    result = means(value, -value)
    for prediction in result.values():
        np.testing.assert_array_equal(prediction, np.zeros_like(value))
    np.testing.assert_allclose(means(value, value)["asinh-mean"], value, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(means(value, value)["physical-mean"], value)


def test_overlap_partition_and_constant_correlation():
    truth = np.array([1e-12, 1e-12, 1e-12, 1e-12])
    first = np.array([1e-12, 1e-12, 2e-12, 2e-12])
    second = np.array([1e-12, 2e-12, 1e-12, 0.])
    result = agreement(first, second, truth, np.ones(4, bool))
    assert [result[key] for key in ("both_pass", "first_only", "second_only", "both_fail")] == [1, 1, 1, 1]
    assert result["oracle_union_pass"] == 3
    assert result["both_fail_opposite"] == 1
    assert agreement(truth, truth, truth, np.ones(4, bool))["correlation"] is None
    assert agreement(truth, truth, truth, np.zeros(4, bool))["components"] == 0
