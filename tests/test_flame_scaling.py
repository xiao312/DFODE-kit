import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.verify_scaling import compare_arrays


def test_scaling_checks_raw_prefix_and_label_mask(tmp_path):
    states = np.array([[300, 101325, .5, .5], [500, 101325, .3, .7], [800, 101325, .2, .8]])
    delta = np.array([[-.01, .01], [-.02, .02], [-.03, .03]])
    np.savez(tmp_path / "small.npz", states=states[:2], snapshot=["a", "b"])
    np.savez(tmp_path / "large.npz", states=states, snapshot=["a", "b", "c"])
    with np.load(tmp_path / "small.npz") as small, np.load(tmp_path / "large.npz") as large:
        first = {"delta": delta[:2], "accepted": np.ones(2, dtype=bool)}
        second = {"delta": delta, "accepted": np.ones(3, dtype=bool)}
        result = compare_arrays(small, large, first, second)
        assert result["state_identity"] and result["label_difference_budget_max"] == 0
        second["accepted"][0] = False
        with pytest.raises(AssertionError):
            compare_arrays(small, large, first, second)
        second["accepted"][0] = True
        second["delta"] = delta.copy()
        second["delta"][0, 0] += 1e-4
        with pytest.raises(ValueError):
            compare_arrays(small, large, first, second)
