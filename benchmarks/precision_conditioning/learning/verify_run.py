"""Read-only verification of saved predictions and the matched-fit protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import network
from targets import decode


def verify(run):
    training = Path(run) / "training"
    summary = json.loads((training / "summary.json").read_text())
    if summary["status"] != "complete" or summary["failures"]:
        raise ValueError("Incomplete comparison")
    torch.set_num_threads(1)
    checked = []
    for mechanism, audit in summary["datasets"].items():
        data = np.load(training / mechanism / "dataset.npz", allow_pickle=False)
        test = np.flatnonzero(data["split"] == "test")
        parent_sets = [set(data["parent"][data["split"] == split]) for split in ("train", "validation", "test")]
        if any(parent_sets[i] & parent_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Parent leakage")
        zero_error = np.abs(data["delta"][test, 1:]) / data["weights"][test, 1:]
        np.testing.assert_allclose(np.quantile(zero_error, .99), audit["zero_baseline"]["budget_p99"], rtol=1e-12)
        variants = [r for r in summary["variants"] if r["mechanism"] == mechanism]
        if len(variants) != 6 or len({r["initial_weights_sha256"] for r in variants}) != 1:
            raise ValueError("Initial weights or variant count mismatch")
        for result in variants:
            destination = training / mechanism / f"{result['target']}-{result['precision']}"
            saved = np.load(destination / "test-predictions.npz", allow_pickle=False)
            np.testing.assert_array_equal(test, saved["indices"])
            errors = np.abs(saved["predicted"][:, 1:] - data["delta"][test, 1:]) / data["weights"][test, 1:]
            p99 = float(np.quantile(errors.ravel(), .99))
            np.testing.assert_allclose(p99, result["test"]["budget_p99"], rtol=1e-12)
            expected_updates = summary["config"]["epochs"] * int(np.ceil(audit["counts"]["train"] / summary["config"]["batch_size"]))
            if result["updates"] != expected_updates:
                raise ValueError("Unequal update budget")
            best = min(result["curves"], key=lambda item: item["validation_budget_p99"])
            if result["selected_epoch"] != best["epoch"]:
                raise ValueError("Epoch selection does not match validation")
            parameters = np.load(destination / "preprocessing.npz", allow_pickle=False)
            dtype = getattr(torch, result["precision"])
            model = network(data["inputs"].shape[1], data["delta"].shape[1], summary["config"]["hidden_widths"], summary["config"]["seed"], dtype)
            model.load_state_dict(torch.load(destination / "weights.pt", weights_only=True))
            x = (data["inputs"][test] - parameters["input_offset"]) / parameters["input_scale"]
            with torch.no_grad():
                predicted = model(torch.as_tensor(x, dtype=dtype)).double().numpy()
            predicted[:, ~parameters["active"]] = 0
            transformed = predicted * parameters["output_scale"]
            scale = data["weights"][test] if result["target"] == "budget-linear" else parameters["asinh_scale"] if result["target"] == "scaled-asinh" else 1.
            replayed = decode(transformed, result["target"], scale)
            np.testing.assert_allclose(replayed, saved["predicted"], rtol=1e-12, atol=0)
            checked.append({"mechanism": mechanism, "target": result["target"], "precision": result["precision"], "recomputed_p99": p99})
    return {"verified_variants": len(checked), "checks": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify(parser.parse_args().output)))
