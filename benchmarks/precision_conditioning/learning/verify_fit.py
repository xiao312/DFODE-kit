"""Read-only replay and independent checks of the training-fit evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import network, preprocessing
from fit_diagnostic import state_hash, subsets


def verify(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "complete" or len(summary["results"]) != 16:
        raise ValueError("Expected a complete 16-control diagnosis")
    config = summary["plan"]["config"]
    if summary["plan"]["test_rows_used"] != 0:
        raise ValueError("Held-out rows must not enter this test")
    hashes, checked = {}, []
    for result in summary["results"]:
        name = result["mechanism"]
        with np.load(run / name / "dataset.npz", allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        if not np.all(data["split"] == "train"):
            raise ValueError("Held-out data in diagnostic arrays")
        directory = run / result["directory"]
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            predicted, indices = saved["predicted"], saved["indices"]
        np.testing.assert_array_equal(indices, subsets(data, name, config)[result["subset"]])
        np.testing.assert_array_equal(data["id"][indices], result["row_ids"])
        with np.load(directory / "preprocessing.npz", allow_pickle=False) as saved:
            parameters = {key: saved[key] for key in saved.files}
        input_norm, output_norm, _, active, _ = preprocessing(data, "budget-linear")
        for expected, actual in [(input_norm.offset, parameters["input_offset"]),
                                 (input_norm.scale, parameters["input_scale"]),
                                 (output_norm.scale, parameters["output_scale"]),
                                 (active, parameters["active"])]:
            np.testing.assert_array_equal(expected, actual)
        model = network(data["inputs"].shape[1], data["delta"].shape[1], config["hidden_widths"], config["seed"], torch.float64)
        if state_hash(model) != result["initial_weights_sha256"]:
            raise ValueError("Initial model differs from pinned seed")
        hashes.setdefault(name, set()).add(result["initial_weights_sha256"])
        model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
        x = torch.as_tensor(input_norm.transform(data["inputs"][indices]), dtype=torch.float64)
        with torch.no_grad():
            normalized = model(x).numpy()
        normalized[:, ~active] = 0
        replayed = normalized * output_norm.scale * data["weights"][indices]
        np.testing.assert_allclose(replayed, predicted, rtol=1e-12, atol=0)
        errors = np.abs(predicted - data["delta"][indices]) / data["weights"][indices]
        independent = {"budget_p99": np.quantile(errors[:, 1:], .99), "budget_max": errors[:, 1:].max(),
                       "budget_rms": np.sqrt(np.mean(errors[:, 1:]**2)),
                       "budget_exceedance": np.mean(errors[:, 1:] > 1), "temperature_budget_max": errors[:, 0].max()}
        for key, value in independent.items():
            np.testing.assert_allclose(value, result["final"][key], rtol=1e-12, atol=0)
        if result["final"]["all_components_pass"] != bool(errors.max() <= 1):
            raise ValueError("Incorrect fit verdict")
        if result["method"] == "adam":
            if result["updates"] != config["updates"] or [row["updates"] for row in result["snapshots"]] != config["snapshots"]:
                raise ValueError("Changed Adam update budget")
        elif result["method"] != "linear-head-svd" or result["updates"] != 0:
            raise ValueError("Unexpected control method")
        checked.append({"run": result["directory"], "p99": float(independent["budget_p99"]), "pass": bool(errors.max() <= 1)})
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Unmatched initial weights")
    return {"verified_models": len(checked), "checks": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    torch.set_num_threads(1)
    print(json.dumps(verify(parser.parse_args().run), allow_nan=False))
