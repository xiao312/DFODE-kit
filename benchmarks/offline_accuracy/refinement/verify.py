"""Reload final weights and independently reconcile physical acceptance counts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.verify_evaluation import check_summary
from .fit import prediction, reload_model
from .plan import configuration
from .run import inputs, save


def verify(dataset, audit_directory, base_directory, directory):
    result = json.loads((directory / "result.json").read_text())
    config = result["config"]
    if result["status"] != "complete" or config != configuration(config["name"], config["seed"]):
        raise ValueError("Require a complete member of the frozen matrix")
    if result["updates_completed"] != config["updates"]:
        raise ValueError("Incomplete update budget")
    training, validation, physics, audit, base, base_result, hashes = inputs(dataset, audit_directory, base_directory, config["seed"])
    if hashes != result["hashes"]:
        raise ValueError("Input or frozen-base identity changed")
    for name, digest in result["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Artifact changed: {name}")
    np.testing.assert_array_equal(np.load(directory / "training-indices.npy"), training["source_indices"])
    model, preprocessing = reload_model(directory, config)
    frozen = base if config["residual"] else None
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrections = prediction(model, preprocessing, rows["states"], config["target"], frozen)
        with np.load(directory / f"{name}-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["source_indices"], rows["source_indices"])
            np.testing.assert_array_equal(saved["correction"], corrections)
            np.testing.assert_array_equal(saved["prediction"], predicted)
        check_summary(predicted, rows["delta"], result[name], physics["species_names"])
        actual = recompute(rows["states"], predicted, rows["delta"], dataset / "mechanism.yaml", physics["interval"])
        assert_scores(actual, result[f"{name}_physical"])
    selected, reference, uncertainty = audit_subset(validation, audit)
    check_summary(predicted[selected], reference, result["audited_subset"], physics["species_names"], uncertainty)
    check_summary(np.zeros_like(validation["delta"]), validation["delta"], result["zero_baseline"], physics["species_names"])
    base_cost = base_result["process_seconds"] if frozen else 0.
    np.testing.assert_equal(result["total_training_process_seconds"], result["training_process_seconds"] + base_cost)
    parameters = sum(p.numel() for p in model.parameters())
    if result["total_parameter_count"] != parameters * (2 if frozen else 1):
        raise ValueError("Parameter cost mismatch")
    return dict(status="verified", result_sha256=sha256(directory / "result.json"),
                model_replay="exact", independent_counts=True, physical_checks=True,
                scope="Development snapshots; subset uncertainty is not a population certificate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Verification output must be new")
    torch.set_num_threads(1)
    result = verify(args.dataset, args.audit, args.base, args.directory)
    if args.output:
        save(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
