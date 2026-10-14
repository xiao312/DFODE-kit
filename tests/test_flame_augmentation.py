import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.flame_conditioning.augmentation import sample_split, validate_config


def fixture():
    config = json.loads((Path(__file__).parents[1] / "benchmarks/flame_conditioning/dataset.json").read_text())
    config.update(train_snapshots=["1"], validation_snapshots=["2"], train_count=32, validation_count=16)
    profile = [[300, 101325, .1, .15, .74, .01],
               [700, 101325, .08, .18, .73, .01],
               [1500, 101325, .04, .2, .75, .01],
               [2200, 101325, .01, .23, .75, .01]]
    source = {"states": np.tile(profile, (2, 1)), "snapshot": np.repeat(["1", "2"], 4),
              "coordinate": np.tile(np.arange(4), 2)}
    return config, source, ["FUEL", "PRODUCT", "N2", "AR"]


def test_valid_config_and_split_lineage():
    config, source, names = fixture()
    validate_config(config, source["snapshot"])
    train, lineage, report = sample_split(source, names, config, "train")
    validation, heldout, _ = sample_split(source, names, config, "validation")
    assert train.shape == (32, 6) and validation.shape == (16, 6)
    assert set(lineage["snapshot"]) == {"1"}
    assert set(heldout["snapshot"]) == {"2"}
    assert max(lineage["right_source_row"]) < min(heldout["left_source_row"])
    np.testing.assert_array_equal(lineage["right_source_row"] - lineage["left_source_row"], 1)
    np.testing.assert_allclose(train[:, 2:].sum(axis=1), 1, rtol=0, atol=4e-16)
    np.testing.assert_allclose(train[:, -1], .01, rtol=0, atol=1e-17)
    assert np.all(train[:, 2:] >= 0)
    assert report["attempts"] >= report["accepted"]


def test_augmentation_is_repeatable_and_does_not_change_source():
    config, source, names = fixture()
    original = source["states"].copy()
    first, _, _ = sample_split(source, names, config, "train")
    second, _, _ = sample_split(source, names, config, "train")
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(source["states"], original)


@pytest.mark.parametrize("key,value", [("validation_snapshots", ["1"]), ("train_count", 0),
                                      ("species_exponent_perturbation", 1), ("wall_seconds", 4000),
                                      ("train_snapshots", ["unknown"])])
def test_bad_config_rejected(key, value):
    config, source, _ = fixture()
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config, source["snapshot"])
