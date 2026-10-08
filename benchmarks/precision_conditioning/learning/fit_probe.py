"""Replay a saved FP64 model on training rows; exit 1 outside species budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import network
from targets import decode


def probe(run, mechanism, target, row=None):
    training = Path(run) / "training"
    config = json.loads((training / "summary.json").read_text())["config"]
    with np.load(training / mechanism / "dataset.npz", allow_pickle=False) as saved:
        indices = np.flatnonzero(saved["split"] == "train")
        if row is not None:
            if not 0 <= row < len(indices):
                raise ValueError("Row must index an existing training row")
            indices = indices[row:row + 1]
        data = {key: saved[key][indices] for key in saved.files}
    directory = training / mechanism / f"{target}-float64"
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as saved:
        parameters = {key: saved[key] for key in saved.files}
    model = network(data["inputs"].shape[1], data["delta"].shape[1],
                    config["hidden_widths"], config["seed"], torch.float64)
    model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True))
    x = (data["inputs"] - parameters["input_offset"]) / parameters["input_scale"]
    with torch.no_grad():
        normalized = model(torch.as_tensor(x, dtype=torch.float64)).numpy()
    normalized[:, ~parameters["active"]] = 0
    scale = data["weights"] if target == "budget-linear" else parameters["asinh_scale"] if target == "scaled-asinh" else 1.
    prediction = decode(normalized * parameters["output_scale"], target, scale)
    error = np.abs(prediction[:, 1:] - data["delta"][:, 1:]) / data["weights"][:, 1:]
    maximum = float(np.max(error))
    return {"mechanism": mechanism, "target": target, "split": "train",
            "rows": len(indices), "components": error.size, "row_ids": data["id"].tolist(),
            "budget_p99": float(np.quantile(error, .99)), "budget_max": maximum,
            "passed": bool(np.isfinite(maximum) and maximum <= 1)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mechanism", choices=["h2", "ch4"], default="h2")
    parser.add_argument("--target", choices=["budget-linear", "signed-power", "scaled-asinh"], default="budget-linear")
    parser.add_argument("--row", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = probe(args.run, args.mechanism, args.target, args.row)
    print(json.dumps(result, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)
