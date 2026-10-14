import numpy as np
import pytest

from benchmarks.flame_conditioning.augmentation import sample_split, validate_config
from tests.test_flame_augmentation import fixture


def test_pressure_sampling_changes_only_pressure_preserves_prefix_and_sources():
    config, source, names = fixture()
    original = source["states"].copy()
    old, lineage, _ = sample_split(source, names, config, "train")
    config["pressure_bounds_Pa"] = [96258.75, 106391.25]
    validate_config(config, source["snapshot"])
    new, new_lineage, _ = sample_split(source, names, config, "train")
    np.testing.assert_array_equal(old[:, 0], new[:, 0])
    np.testing.assert_array_equal(old[:, 2:], new[:, 2:])
    np.testing.assert_array_equal(original, source["states"])
    for key in lineage:
        np.testing.assert_array_equal(lineage[key], new_lineage[key])
    assert np.all((new[:, 1] >= 96258.75) & (new[:, 1] <= 106391.25))
    assert np.ptp(new[:, 1]) > 5000
    config["train_count"] = 64
    expanded, _, _ = sample_split(source, names, config, "train")
    np.testing.assert_array_equal(new, expanded[:32])


@pytest.mark.parametrize("bounds", [[0, 1], [2, 1], [1], [1, np.nan]])
def test_invalid_pressure_bounds_rejected(bounds):
    config, source, _ = fixture()
    config["pressure_bounds_Pa"] = bounds
    with pytest.raises(ValueError, match="pressure_bounds_Pa"):
        validate_config(config, source["snapshot"])
