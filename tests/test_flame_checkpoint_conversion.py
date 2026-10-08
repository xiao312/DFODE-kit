from collections import OrderedDict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from benchmarks.flame_conditioning.checkpoint_conversion.convert import SHAPES, checked_arrays, numeric_array


def test_numeric_array_rejects_objects_nonfinite_and_wrong_shape():
    value = np.array([1., 2.], dtype=np.float32)
    np.testing.assert_array_equal(numeric_array(value, (2,), "value"), value)
    for invalid in (np.array([object(), object()]), np.array([1., np.nan]), np.ones(3), [1., 2.]):
        with pytest.raises(ValueError):
            numeric_array(invalid, (2,), "value")


def test_conversion_enforces_exact_parameter_contract():
    weights = OrderedDict()
    for index, shape in enumerate(SHAPES):
        weights[f"net.linear_layer_{index}.weight"] = torch.zeros(shape)
        weights[f"net.linear_layer_{index}.bias"] = torch.zeros(shape[0])
    stats = {"features_mean": torch.zeros(61), "features_std": torch.ones(61),
             "labels_mean": torch.zeros(58), "labels_std": torch.ones(58)}
    payload = {"model_state_dict": weights, "normalization_stats": stats}
    assert len(checked_arrays(payload, "signed-power")) == 14
    stats["features_std"][0] = 0
    with pytest.raises(ValueError, match="Nonpositive"):
        checked_arrays(payload, "signed-power")
    stats["features_std"][0] = 1
    stats["labels_mean"][0] = 1
    with pytest.raises(ValueError, match="zero direct-power"):
        checked_arrays(payload, "signed-power")
