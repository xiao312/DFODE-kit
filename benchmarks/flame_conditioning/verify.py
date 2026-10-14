"""Read-only replay of each saved flame model and its matched-budget evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.coordinates import decode, input_features
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.train import network
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores


def load_predictor(directory, config):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    preprocessing = dict(np.load(directory / "preprocessing.npz", allow_pickle=False))
    dtype = getattr(torch, result["precision"])
    model = network(len(preprocessing["x_scale"]), len(preprocessing["y_scale"]),
                    config["hidden_widths"], config["seed"], dtype, config.get("activation", "tanh"))
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()

    def predict(states):
        values = (input_features(states) - preprocessing["x_offset"]) / preprocessing["x_scale"]
        with torch.no_grad():
            encoded = model(torch.as_tensor(values, dtype=dtype)).double().numpy()
        encoded = encoded * preprocessing["y_scale"] + preprocessing["y_offset"]
        predicted, corrected = decode(states[:, 2:], encoded, result["target"], preprocessing["asinh_scale"])
        predicted[:, ~preprocessing["active"]] = 0
        corrected[:, ~preprocessing["active"]] = False
        return predicted, corrected

    return predict, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("--training-metrics", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    torch.set_num_threads(1)
    data, physics, _ = load_dataset(args.dataset)
    summary = json.loads((args.training / "summary.json").read_text())
    if summary["dataset_manifest_sha256"] != sha256(args.dataset / "manifest.json"):
        raise ValueError("Training data provenance mismatch")
    config = summary["plan"]["config"]
    expected = len(config["targets"]) * len(config["precisions"]) * len(config["training_sizes"])
    if summary["status"] != "complete" or len(summary["variants"]) != expected:
        raise ValueError("Cannot verify a complete matched comparison from partial training")
    validation = data["validation"]
    active = np.array([name != "AR" for name in physics["species_names"]])
    permutation = np.random.default_rng(config["seed"]).permutation(len(data["train"]["states"]))
    replayed, initial_hashes = [], set()
    for variant in summary["variants"]:
        directory = args.training / variant["name"]
        predict, result = load_predictor(directory, config)
        predicted, corrections = predict(validation["states"])
        saved = np.load(directory / "validation-predictions.npz", allow_pickle=False)
        np.testing.assert_array_equal(saved["source_indices"], validation["source_indices"])
        np.testing.assert_array_equal(saved["correction"], corrections)
        np.testing.assert_allclose(saved["prediction"], predicted, rtol=1e-13, atol=1e-30)
        size = min(variant["requested_training_count"], len(permutation))
        expected_indices = data["train"]["source_indices"][permutation[:size]]
        np.testing.assert_array_equal(np.load(directory / "training-indices.npy"), expected_indices)
        weighted = np.abs(predicted - validation["delta"]) / (1e-12 + 1e-6 * np.abs(validation["states"][:, 2:]))
        p99 = float(np.quantile(weighted[:, active], .99))
        np.testing.assert_allclose(p99, result["validation"]["budget_error"]["p99"], rtol=1e-12)
        if result["updates_completed"] != config["updates"] or result["status"] != "complete":
            raise ValueError("Variant has an unequal update budget")
        initial = network(validation["states"].shape[1], validation["delta"].shape[1],
                          config["hidden_widths"], config["seed"], getattr(torch, result["precision"]), config.get("activation", "tanh"))
        initial_hash = hashlib.sha256(b"".join(parameter.detach().double().numpy().tobytes() for parameter in initial.parameters())).hexdigest()
        if initial_hash != result["initial_weights_sha256"]:
            raise ValueError("Initialization hash mismatch")
        initial_hashes.add(initial_hash)
        record = {"name": variant["name"], "validation_budget_p99": p99, "training_count": size}
        if args.training_metrics:
            selected = permutation[:size]
            training_states = data["train"]["states"][selected]
            training_prediction, _ = predict(training_states)
            actual = recompute(training_states, training_prediction, data["train"]["delta"][selected],
                               args.dataset / "mechanism.yaml", physics["interval"])
            assert_scores(actual, result["training"])
            record["training_metrics"] = actual
        replayed.append(record)
    if len(initial_hashes) != 1:
        raise ValueError("Target/precision variants did not start from matching weights")
    report = {"status": "verified", "models": len(replayed), "replayed": replayed,
              "training_metrics_checked": args.training_metrics,
              "training_summary_sha256": sha256(args.training / "summary.json")}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
