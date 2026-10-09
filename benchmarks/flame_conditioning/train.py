"""Matched bounded flame learning curves; never open the reserved 2D test."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.coordinates import KINDS, decode, encode, input_features, standardization
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores


def network(inputs, outputs, widths, seed, dtype, activation="tanh"):
    activations = {"tanh": torch.nn.Tanh, "gelu": torch.nn.GELU}
    if activation not in activations:
        raise ValueError("Unsupported activation")
    torch.manual_seed(seed)
    layers = []
    for width in widths:
        layers.extend([torch.nn.Linear(inputs, width), activations[activation]()])
        inputs = width
    layers.append(torch.nn.Linear(inputs, outputs))
    return torch.nn.Sequential(*layers).to(dtype=dtype)


def validate_config(config):
    if config.get("checkpoint_selection", "validation-p99") not in ("validation-p99", "final"):
        raise ValueError("checkpoint_selection must be validation-p99 or final")
    if "pressure_bounds_Pa" in config:
        bounds = np.asarray(config["pressure_bounds_Pa"], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("pressure_bounds_Pa must be two increasing positive finite bounds")
    if config.get("activation", "tanh") not in ("tanh", "gelu") or config.get("loss", "mse") not in ("mse", "l1"):
        raise ValueError("Unsupported activation or loss")
    if config["schema_version"] != 1 or not set(config["targets"]) <= set(KINDS):
        raise ValueError("Unsupported training schema or target")
    if not config["targets"] or len(set(config["targets"])) != len(config["targets"]):
        raise ValueError("Targets must be nonempty and unique")
    if not config["precisions"] or not set(config["precisions"]) <= {"float32", "float64"}:
        raise ValueError("Supported precisions are float32 and float64")
    for name in ("updates", "batch_size", "validation_every", "wall_seconds", "maximum_variant_seconds"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["wall_seconds"] > 3600 or config["updates"] > 20000:
        raise ValueError("Configuration exceeds bounded run limits")
    if any(type(width) is not int or not 1 <= width <= 800 for width in config["hidden_widths"]):
        raise ValueError("Invalid hidden width")
    if not config["training_sizes"] or any(type(size) is not int or size < 32 for size in config["training_sizes"]):
        raise ValueError("Training sizes must be integers >=32")
    for name in ("learning_rate", "final_learning_rate"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError("Learning rates must be finite and positive")


def fit_variant(training, validation, physics, config, target, precision, destination, deadline):
    destination.mkdir()
    started = time.monotonic()
    process_started = time.process_time()
    dtype = getattr(torch, precision)
    states, delta = training["states"], training["delta"]
    x_offset, x_scale = standardization(input_features(states))
    if "pressure_bounds_Pa" in config:
        lower, upper = config["pressure_bounds_Pa"]
        if any(np.any((rows["states"][:, 1] < lower) | (rows["states"][:, 1] > upper)) for rows in (training, validation)):
            raise ValueError("Training/evaluation pressure is outside the declared offline domain")
        x_offset[1], x_scale[1] = (lower + upper) / 2, (upper - lower) / 2
    asinh_scale = np.maximum(np.quantile(np.abs(delta), .9, axis=0), 1e-32)
    transformed = encode(states[:, 2:], delta, target, asinh_scale)
    y_offset, y_scale = standardization(transformed)
    active = np.any(delta != 0, axis=0) & np.array([name != "AR" for name in physics["species_names"]])
    if not active.any():
        raise ValueError("No active output species")
    x_train = torch.as_tensor((input_features(states) - x_offset) / x_scale, dtype=dtype)
    x_val = torch.as_tensor((input_features(validation["states"]) - x_offset) / x_scale, dtype=dtype)
    y_train = torch.as_tensor((transformed - y_offset) / y_scale, dtype=dtype)
    active_tensor = torch.as_tensor(active)
    model = network(x_train.shape[1], delta.shape[1], config["hidden_widths"], config["seed"], dtype, config.get("activation", "tanh"))
    initial_hash = hashlib.sha256(b"".join(p.detach().double().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    best_score, best_state, selected_step = float("inf"), None, None
    history, status, step = [], "complete", 0

    def predict(features, initial):
        with torch.no_grad():
            coordinate = model(features).double().numpy() * y_scale + y_offset
        prediction, correction = decode(initial[:, 2:], coordinate, target, asinh_scale)
        prediction[:, ~active] = 0
        correction[:, ~active] = False
        return prediction, correction

    for step in range(1, config["updates"] + 1):
        if time.monotonic() >= deadline:
            status = "time_limit"
            step -= 1
            break
        progress = (step - 1) / max(config["updates"] - 1, 1)
        rate = config["final_learning_rate"] + .5 * (config["learning_rate"] - config["final_learning_rate"]) * (1 + np.cos(np.pi * progress))
        optimizer.param_groups[0]["lr"] = rate
        indices = rng.integers(0, len(states), size=config["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        difference = model(x_train[indices])[:, active_tensor] - y_train[indices][:, active_tensor]
        loss = difference.abs().mean() if config.get("loss", "mse") == "l1" else difference.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        if step % config["validation_every"] == 0 or step == config["updates"]:
            try:
                predicted, corrections = predict(x_val, validation["states"])
                weighted = np.abs(predicted - validation["delta"]) / (1e-12 + 1e-6 * np.abs(validation["states"][:, 2:]))
                score = float(np.quantile(weighted[:, active], .99))
                history.append({"step": step, "training_batch_loss": float(loss.item()),
                                "validation_budget_p99": score,
                                "inverse_domain_correction_fraction": float(corrections[:, active].mean())})
                use_final = config.get("checkpoint_selection") == "final"
                if (use_final and step == config["updates"]) or (not use_final and score < best_score):
                    best_score, selected_step, best_state = score, step, copy.deepcopy(model.state_dict())
            except ValueError as error:
                history.append({"step": step, "validation_error": str(error)})
    if best_state is None:
        raise ValueError("No finite validation checkpoint within the compute limit")
    model.load_state_dict(best_state)
    predicted, corrections = predict(x_val, validation["states"])
    training_prediction, training_corrections = predict(x_train, states)
    result = {
        "status": status, "target": target, "precision": precision, "training_count": len(states),
        "updates_completed": step, "updates_planned": config["updates"], "selected_step": selected_step,
        "initial_weights_sha256": initial_hash, "elapsed_seconds": time.monotonic() - started,
        "process_seconds": time.process_time() - process_started,
        "inactive_species": [name for name, enabled in zip(physics["species_names"], active, strict=True) if not enabled],
        "validation": physical_scores(predicted, validation["delta"], validation["states"], corrections, **physics),
        "training": physical_scores(training_prediction, delta, states, training_corrections, **physics),
        "history": history,
    }
    torch.save(best_state, destination / "weights.pt")
    np.savez(destination / "preprocessing.npz", x_offset=x_offset, x_scale=x_scale,
             y_offset=y_offset, y_scale=y_scale, asinh_scale=asinh_scale, active=active)
    np.savez_compressed(destination / "validation-predictions.npz", prediction=predicted, correction=corrections,
                        source_indices=validation["source_indices"])
    np.save(destination / "training-indices.npy", training["source_indices"])
    (destination / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("learning.json"))
    parser.add_argument("--audit", type=Path, required=True, help="Completed augmented-label audit directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory")
    config = json.loads(args.config.read_text())
    validate_config(config)
    data, physics, dataset_manifest = load_dataset(args.dataset)
    audit = json.loads((args.audit / "summary.json").read_text())
    if (audit.get("status") != "complete" or audit.get("reference_subset_pass") is not True
            or audit.get("dataset_manifest_sha256") != sha256(args.dataset / "manifest.json")):
        raise ValueError("Require a passing augmented-label audit for this exact dataset")
    plan = {"config": config, "available": {split: len(rows["states"]) for split, rows in data.items()},
            "test": "No reserved 2D test data are loaded or scored"}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    if torch.__version__ != "2.5.1+cpu":
        raise ValueError(f"Expected pinned torch 2.5.1+cpu, found {torch.__version__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    summary = {"status": "running", "plan": plan, "source": source_revision(), "torch": torch.__version__,
               "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
               "audit_summary_sha256": sha256(args.audit / "summary.json"),
               "variants": [], "failures": [], "dataset_source": dataset_manifest["source"]}
    validation = data["validation"]
    summary["zero_baseline"] = physical_scores(np.zeros_like(validation["delta"]), validation["delta"],
        validation["states"], np.zeros_like(validation["delta"], dtype=bool), **physics)

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    permutation = np.random.default_rng(config["seed"]).permutation(len(data["train"]["states"]))
    save()
    for requested_size in config["training_sizes"]:
        selected = permutation[:min(requested_size, len(permutation))]
        training = {key: values[selected] for key, values in data["train"].items()}
        for precision in config["precisions"]:
            for target in config["targets"]:
                name = f"n{requested_size}-{precision}-{target}"
                deadline = min(started + config["wall_seconds"], time.monotonic() + config["maximum_variant_seconds"])
                if time.monotonic() >= deadline:
                    summary["status"] = "time_limit"
                    save()
                    return
                try:
                    result = fit_variant(training, validation, physics, config, target, precision, args.output / name, deadline)
                    result.update(name=name, requested_training_count=requested_size)
                    summary["variants"].append(result)
                    print(json.dumps({"name": name, "status": result["status"], "updates": result["updates_completed"],
                                      "validation_budget_p99": result["validation"]["budget_error"]["p99"]}), flush=True)
                except Exception as error:
                    summary["failures"].append({"name": name, "error": str(error)})
                    print(json.dumps({"name": name, "error": str(error)}), flush=True)
                save()
    summary["status"] = "complete" if not summary["failures"] and all(v["status"] == "complete" for v in summary["variants"]) else "incomplete"
    save()


if __name__ == "__main__":
    main()
