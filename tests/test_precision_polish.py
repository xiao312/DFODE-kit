"""Strict pass/save and deterministic polish controls; synthetic data only."""
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/precision_conditioning/learning"))


def core():
    pytest.importorskip("torch")
    pytest.importorskip("cantera")
    import polish_core
    return polish_core


def test_temperature_and_nonfinite_error_cannot_pass():
    module = core()
    reference = np.zeros((1, 3))
    weights = np.ones((1, 3))
    assert module.budget_max(np.array([[2., 0., 0.]]), reference, weights) == 2
    with pytest.raises(ValueError, match="finite"):
        module.budget_max(np.array([[np.nan, 0., 0.]]), reference, weights)


def test_pass_stops_before_another_update_and_saves_weights(tmp_path):
    module = core()
    data = {"id": np.array(["a", "b"]), "split": np.array(["train", "train"]),
            "inputs": np.array([[0., 1.], [1., 1.]]), "delta": np.array([[1., .1], [2., .2]]),
            "weights": np.ones((2, 2)), "initial": np.ones((2, 2)), "relative_mask": np.ones((2, 2), dtype=bool)}
    config = {"seed": 42, "hidden_widths": [4, 4], "warmup_updates": 0, "learning_rate": .001,
              "adam_updates": 10, "adam_final_rate": 1e-7, "lbfgs_steps": 10, "lbfgs_history": 5,
              "stop_budget_max": .5, "fit_seconds": 30}
    result = module.fit(data, np.arange(2), config, "linear-head", tmp_path / "fit", time.monotonic() + 30)
    assert result["stop_reason"] == "budget_pass"
    assert result["selected"]["max_budget"] <= .5
    assert result["final"]["all_components_pass"]
    assert result["history"][-1]["max_budget"] <= .5
    assert sum(row["max_budget"] <= .5 for row in result["history"]) == 1
    assert (tmp_path / "fit/weights.pt").is_file()


def test_expired_fit_keeps_initial_model_and_reports_timeout(tmp_path):
    module = core()
    data = {"id": np.array(["a"]), "split": np.array(["train"]), "inputs": np.array([[0., 1.]]),
            "delta": np.array([[1., .1]]), "weights": np.full((1, 2), 1e-9),
            "initial": np.ones((1, 2)), "relative_mask": np.ones((1, 2), dtype=bool)}
    config = {"seed": 42, "hidden_widths": [4, 4], "warmup_updates": 2, "learning_rate": .001,
              "stop_budget_max": .5, "fit_seconds": 30}
    result = module.fit(data, np.array([0]), config, "adam-decay", tmp_path / "fit", time.monotonic() - 1)
    assert result["stop_reason"] == "time_limit"
    assert result["selected"]["stage"] == "initial"
    assert not result["final"]["all_components_pass"]


@pytest.mark.parametrize("method", ["adam-decay", "lbfgs", "linear-head"])
def test_initial_pass_prevents_all_updates(method, tmp_path):
    module = core()
    data = {"id": np.array(["a"]), "split": np.array(["train"]), "inputs": np.array([[0., 1.]]),
            "delta": np.array([[1., .1]]), "weights": np.full((1, 2), 1e6),
            "initial": np.ones((1, 2)), "relative_mask": np.ones((1, 2), dtype=bool)}
    config = {"seed": 42, "hidden_widths": [4, 4], "warmup_updates": 200, "learning_rate": .001,
              "stop_budget_max": .5, "fit_seconds": 30}
    result = module.fit(data, np.array([0]), config, method, tmp_path / "fit", time.monotonic() + 30)
    assert result["stop_reason"] == "budget_pass"
    assert len(result["history"]) == 1
    assert result["initial_weights_sha256"] == result["selected_weights_sha256"]


def test_public_summary_keeps_selection_but_excludes_full_history():
    from polish_review import public_summary
    source = {"status": "complete", "results": [{"selected": {"max_budget": .1},
              "history": [{"max_budget": 2.}, {"max_budget": .1}]}]}
    exported = public_summary(source)
    assert "history" not in exported["results"][0]
    assert exported["results"][0]["history_points"] == 2
    assert exported["results"][0]["selected"]["max_budget"] == .1
    assert len(source["results"][0]["history"]) == 2
