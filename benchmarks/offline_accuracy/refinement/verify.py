"""Reload final weights and independently reconcile physical acceptance counts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.coordinates import input_features, standardization
from benchmarks.flame_conditioning.train import network
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.verify_evaluation import check_summary
from .coordinates import encode
from .arithmetic import residual_round_trip
from .fit import prediction, reload_model, weight_hash
from .plan import COORDINATE_SCALE, configuration
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
    initial = network(training["states"].shape[1], training["delta"].shape[1],
                      config["widths"], config["seed"], torch.float32, "gelu")
    if weight_hash(initial) != result["initial_weights_sha256"]:
        raise ValueError("Initialization does not match the declared seed")
    check_preprocessing(training, physics["species_names"], config, preprocessing, frozen, result)
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
    timing = result["inference"]
    if timing["states"] != len(validation["states"]) or len(timing["repeats"]) != 5:
        raise ValueError("Inference timing population or repeat count differs")
    for key in ("wall_seconds", "process_seconds"):
        durations = [row[key] for row in timing["repeats"]]
        if not np.isfinite(durations).all() or min(durations) <= 0:
            raise ValueError("Invalid inference duration")
        np.testing.assert_equal(timing[f"median_{key}"], np.median(durations))
    arithmetic = None
    if frozen:
        original, _ = frozen(validation["states"])
        arithmetic = residual_round_trip(original, validation["delta"], physics["species_names"])
    return dict(status="verified", result_sha256=sha256(directory / "result.json"),
                model_replay="exact", independent_counts=True, physical_checks=True,
                training_only_scales_checked=True, residual_arithmetic=arithmetic,
                scope="Development snapshots; subset uncertainty is not a population certificate")


def check_preprocessing(training, species, config, prep, base, result):
    """Rebuild every scale from training rows; validation rows cannot set scales."""
    states, delta = training["states"], training["delta"]
    offset, scale = standardization(input_features(states))
    offset[1], scale[1] = 101325., 5066.25
    np.testing.assert_array_equal(prep["x_offset"], offset)
    np.testing.assert_array_equal(prep["x_scale"], scale)
    active = np.array([name != "AR" for name in species]) & np.any(delta != 0, axis=0)
    np.testing.assert_array_equal(prep["active"], active)
    labels = delta.copy()
    if base:
        predicted, _ = base(states)
        if hashlib.sha256(predicted.tobytes()).hexdigest() != result["frozen_base_training_prediction_sha256"]:
            raise ValueError("Frozen base prediction differs")
        labels -= predicted
        target_scale = np.maximum(np.sqrt(np.mean(labels**2, axis=0)), 1e-30)
    else:
        target_scale = np.maximum(np.quantile(abs(delta), .9, axis=0), 1e-32)
    np.testing.assert_array_equal(prep["target_scale"], target_scale)
    target = config["target"]
    if target in ("budget-log", "budget-asinh", "residual"):
        y_offset = np.zeros(delta.shape[1])
        y_scale = np.full(delta.shape[1], 1. if base else COORDINATE_SCALE)
    else:
        y_offset, y_scale = standardization(encode(states[:, 2:], labels, target, target_scale))
    np.testing.assert_array_equal(prep["y_offset"], y_offset)
    np.testing.assert_array_equal(prep["y_scale"], y_scale)


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
