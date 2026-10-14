"""Replay every selected fit; optionally enforce one method's strict narrow gate."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from fit_diagnostic import state_hash, subsets
from train import network, preprocessing


def verify(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    config = summary["config"]
    expected = set(itertools.product(("h2", "ch4"), ("one", "narrow", "all"), config["seeds"], config["methods"]))
    actual = [(r["mechanism"], r["subset"], r["seed"], r["method"]) for r in summary["results"]]
    if summary["status"] != "complete" or set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate fits; a complete matrix is required")
    if summary["test_rows_used"] != 0:
        raise ValueError("Held-out rows entered the fit test")
    starts, checked = {}, []
    for result in summary["results"]:
        name = result["mechanism"]
        with np.load(run / name / "dataset.npz", allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        indices = subsets(data, name, config)[result["subset"]]
        np.testing.assert_array_equal(data["id"][indices], result["row_ids"])
        directory = run / result["directory"]
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            predicted = saved["predicted"]
            np.testing.assert_array_equal(saved["indices"], indices)
        with np.load(directory / "preprocessing.npz", allow_pickle=False) as saved:
            parameters = {key: saved[key] for key in saved.files}
        input_norm, output_norm, _, active, _ = preprocessing(data, "budget-linear")
        for expected_value, saved_value in [(input_norm.offset, parameters["input_offset"]),
                                           (input_norm.scale, parameters["input_scale"]),
                                           (output_norm.scale, parameters["output_scale"]),
                                           (active, parameters["active"])]:
            np.testing.assert_array_equal(expected_value, saved_value)
        model = network(data["inputs"].shape[1], data["delta"].shape[1], config["hidden_widths"], result["seed"], torch.float64)
        if state_hash(model) != result["initial_weights_sha256"]:
            raise ValueError("Initialization hash mismatch")
        start_key = (name, result["subset"], result["seed"])
        if "warmup_weights_sha256" in result:
            starts.setdefault(start_key, set()).add(result["warmup_weights_sha256"])
        model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
        if state_hash(model) != result["selected_weights_sha256"]:
            raise ValueError("Saved model hash mismatch")
        with torch.no_grad():
            values = model(torch.as_tensor(input_norm.transform(data["inputs"][indices]), dtype=torch.float64)).numpy()
        values[:, ~active] = 0
        replayed = values * output_norm.scale * data["weights"][indices]
        np.testing.assert_allclose(replayed, predicted, rtol=1e-12, atol=0)
        errors = np.abs(replayed - data["delta"][indices]) / data["weights"][indices]
        if not np.all(np.isfinite(errors)):
            raise ValueError("Nonfinite replay error")
        independent = {"budget_max": errors[:, 1:].max(), "budget_p99": np.quantile(errors[:, 1:], .99),
                       "budget_rms": np.sqrt(np.mean(errors[:, 1:]**2)),
                       "budget_exceedance": np.mean(errors[:, 1:] > 1), "temperature_budget_max": errors[:, 0].max()}
        for key, value in independent.items():
            np.testing.assert_allclose(value, result["final"][key], rtol=1e-12, atol=0)
        np.testing.assert_allclose(errors.max(), result["selected"]["max_budget"], rtol=1e-12, atol=0)
        if result["selected"] != min(result["history"], key=lambda point: point["max_budget"]):
            raise ValueError("Selected model is not the first lowest-maximum checkpoint")
        for point in result["history"]:
            np.testing.assert_equal(point["max_budget"], max(point["species_max"], point["temperature_max"]))
        passing = [i for i, point in enumerate(result["history"]) if point["max_budget"] <= config["stop_budget_max"]]
        if passing and (passing != [len(result["history"]) - 1] or result["stop_reason"] != "budget_pass"):
            raise ValueError("A passing fit was not stopped immediately")
        if result["stop_reason"] == "budget_pass" and not passing:
            raise ValueError("Incorrect stop verdict")
        passed = bool(errors.max() <= 1)
        if passed != result["final"]["all_components_pass"]:
            raise ValueError("Incorrect physical pass verdict")
        checked.append({"mechanism": name, "subset": result["subset"], "seed": result["seed"],
                        "method": result["method"], "max_budget": float(errors.max()), "passed": passed})
    if any(len(hashes) != 1 for hashes in starts.values()):
        raise ValueError("Methods started from different warm-up weights")
    gates = {method: all(r["passed"] for r in checked if r["method"] == method and r["subset"] != "all")
             for method in config["methods"]}
    return {"verified_models": len(checked), "narrow_gates": gates, "checks": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--require-narrow", choices=["adam-decay", "lbfgs", "linear-head"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = verify(args.run)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(1 if args.require_narrow and not result["narrow_gates"][args.require_narrow] else 0)
