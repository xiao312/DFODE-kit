"""Bounded experiment invariants; synthetic fixtures do not run chemistry."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

MODULE = Path(__file__).parents[1] / "benchmarks/precision_conditioning/learning"
sys.path.insert(0, str(MODULE))
from dataset import partition
from metrics import error_summary


def fixture():
    config = json.loads((MODULE.parent / "reference/pilot.json").read_text())
    manifest = {"config": config, "mechanisms": [{"id": "h2", "species": ["A", "B"], "species_count": 2}]}
    experiment = {"split_by_temperature": {"900": "train", "1000": "validation", "1100": "test"},
                  "minimum_accepted": {"train": 1, "validation": 1, "test": 1}}
    records = []
    for temperature in (900, 1000, 1100):
        records.append({"id": str(temperature), "trajectory_id": f"h2-{temperature}K", "mechanism": "h2",
                        "state": {"T": temperature, "P": 101325., "Y": [.1, .9]}, "interval_s": 1e-6,
                        "failures": {}, "solutions": {name: {"delta": [1e-6, -1e-8, 1e-8]} for name in
                            ("absolute18", "absolute21", "step_limited", "radau_check", "radau_reference")}})
    return records, manifest, experiment


def test_partition_rejects_duplicate_ids_and_unknown_parents():
    records, manifest, experiment = fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        partition(records + [records[0]], manifest, experiment, "h2")
    records[0]["trajectory_id"] = "unknown"
    with pytest.raises(ValueError, match="Unknown parent"):
        partition(records, manifest, experiment, "h2")


def test_uncertain_interval_excluded_without_changing_threshold():
    records, manifest, experiment = fixture()
    invalid = copy.deepcopy(records[0])
    invalid["id"] = "uncertain"
    invalid["solutions"]["absolute18"]["delta"][1] = 1e-2
    arrays, audit = partition(records + [invalid], manifest, experiment, "h2")
    assert len(arrays["delta"]) == 3
    assert audit["excluded"][0]["reason"] == "reference_disagreement"
    assert audit["counts"] == {"train": 1, "validation": 1, "test": 1}


def test_metrics_use_physical_budget_and_separate_relative_mask():
    result = error_summary(np.array([2., 0.]), np.array([1., 0.]), np.array([.5, .5]), np.array([True, False]))
    assert result["budget_max"] == 2.
    assert result["budget_exceedance"] == .5
    assert result["relative_count"] == 1
    assert result["relative_p99"] == 1.


def test_review_escapes_table_content():
    spec = importlib.util.spec_from_file_location("learning_review", MODULE / "review.py")
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    assert "&lt;script&gt;" in review.table(["Name"], [["<script>"]])


def load_trainer():
    pytest.importorskip("torch")
    pytest.importorskip("cantera")
    spec = importlib.util.spec_from_file_location("precision_learner", MODULE / "train.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_train_only_scales_and_identical_precision_initialization():
    trainer = load_trainer()
    records, manifest, experiment = fixture()
    data, _ = partition(records, manifest, experiment, "h2")
    fitted = trainer.preprocessing(data, "scaled-asinh")
    changed = {key: value.copy() for key, value in data.items()}
    changed["inputs"][1:] *= 100
    changed["delta"][1:] *= 100
    repeated = trainer.preprocessing(changed, "scaled-asinh")
    np.testing.assert_array_equal(fitted[0].offset, repeated[0].offset)
    np.testing.assert_array_equal(fitted[1].scale, repeated[1].scale)
    np.testing.assert_array_equal(fitted[2], repeated[2])
    first = trainer.network(5, 3, [4, 4], 123, trainer.torch.float32)
    second = trainer.network(5, 3, [4, 4], 123, trainer.torch.float64)
    for a, b in zip(first.parameters(), second.parameters()):
        np.testing.assert_array_equal(a.detach().double().numpy(), b.detach().numpy())


def test_small_training_run_has_fixed_updates_and_saved_predictions(tmp_path):
    trainer = load_trainer()
    trainer.torch.set_num_threads(1)
    records, manifest, experiment = fixture()
    data, _ = partition(records, manifest, experiment, "h2")
    config = {"hidden_widths": [4, 4], "seed": 123, "learning_rate": .001, "epochs": 2, "batch_size": 256}
    result = trainer.fit_variant(data, config, "budget-linear", "float64", tmp_path / "model", time.monotonic() + 30)
    assert result["updates"] == 2
    assert result["selected_epoch"] in (1, 2)
    assert len(result["curves"]) == 2
    assert result["test"]["count"] == 2
    assert (tmp_path / "model/test-predictions.npz").is_file()
