import numpy as np
import pytest
import json
from pathlib import Path

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


def test_growth_configs_change_counts_not_the_scientific_contract():
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "flame_conditioning"
    small = json.loads((root / "dataset-50k.json").read_text())
    large = json.loads((root / "dataset-200k.json").read_text())
    assert small.pop("train_count") == 50000
    assert large.pop("train_count") == 200000
    assert small == large
    first = json.loads((root / "learning-source-longer.json").read_text())
    second = json.loads((root / "learning-source-longer200k.json").read_text())
    assert first.pop("training_sizes") == [50000]
    assert second.pop("training_sizes") == [200000]
    first.pop("selection")
    second.pop("selection")
    assert first == second
