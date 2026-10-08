"""Matched CPU training; validation selects epochs and test is scored once."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from dataset import load_run
from metrics import error_summary, evaluate
from targets import TrainOnlyNormalizer, decode, encode, fit_asinh_scale
from pilot import revision


def network(inputs, outputs, widths, seed, dtype):
    torch.manual_seed(seed)
    layers = []
    for width in widths:
        layers.extend([torch.nn.Linear(inputs, width), torch.nn.Tanh()])
        inputs = width
    layers.append(torch.nn.Linear(inputs, outputs))
    return torch.nn.Sequential(*layers).to(dtype=dtype)


def preprocessing(data, target):
    split, delta = data["split"], data["delta"]
    inputs = TrainOnlyNormalizer.fit(data["inputs"], split, center=True)
    asinh_scale = fit_asinh_scale(delta, split)
    scale = data["weights"] if target == "budget-linear" else asinh_scale if target == "scaled-asinh" else 1.
    encoded = encode(delta, target, scale)
    outputs = TrainOnlyNormalizer.fit(encoded, split)
    active = np.any(delta[split == "train"] != 0, axis=0)
    return inputs, outputs, asinh_scale, active, encoded


def fit_variant(data, config, target, precision, output, deadline):
    start = time.perf_counter()
    dtype = getattr(torch, precision)
    input_norm, output_norm, asinh_scale, active, encoded = preprocessing(data, target)
    x = torch.as_tensor(input_norm.transform(data["inputs"]), dtype=dtype)
    y = torch.as_tensor(output_norm.transform(encoded), dtype=dtype)
    indices = {label: np.flatnonzero(data["split"] == label) for label in ("train", "validation", "test")}
    active_tensor = torch.as_tensor(active)
    model = network(x.shape[1], y.shape[1], config["hidden_widths"], config["seed"], dtype)
    initial_hash = hashlib.sha256(b"".join(p.detach().double().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    generator = np.random.default_rng(config["seed"])
    best, best_state, selected_epoch, updates = float("inf"), None, None, 0
    curves = []

    def predict(selected):
        with torch.no_grad():
            normalized = model(x[selected]).double().numpy()
        normalized[:, ~active] = 0
        transformed = output_norm.inverse(normalized)
        scale = data["weights"][selected] if target == "budget-linear" else asinh_scale if target == "scaled-asinh" else 1.
        return decode(transformed, target, scale)

    for epoch in range(1, config["epochs"] + 1):
        if time.monotonic() > deadline:
            raise TimeoutError("Training reached its wall-time limit")
        order = generator.permutation(indices["train"])
        total = 0.
        for begin in range(0, len(order), config["batch_size"]):
            batch = order[begin:begin + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x[batch])[:, active_tensor] - y[batch][:, active_tensor])**2).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
            updates += 1
        val = indices["validation"]
        prediction = predict(val)
        score = error_summary(prediction[:, 1:], data["delta"][val, 1:], data["weights"][val, 1:],
                              data["relative_mask"][val, 1:])["budget_p99"]
        curves.append({"epoch": epoch, "train_coordinate_mse": total / len(order), "validation_budget_p99": score})
        if score < best:
            best, selected_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    test = indices["test"]
    prediction = predict(test)
    # Repeated fixed test batch, including decode; no data-dependent tuning.
    for _ in range(5):
        predict(test)
    timing_start = time.perf_counter()
    for _ in range(50):
        predict(test)
    inference_us = (time.perf_counter() - timing_start) * 1e6 / (50 * len(test))
    result = {"target": target, "precision": precision, "selected_epoch": selected_epoch,
              "validation_budget_p99": best, "updates": updates, "training_seconds": training_seconds,
              "inference_us_per_sample_batched": inference_us, "initial_weights_sha256": initial_hash,
              "inactive_zero_channels": np.flatnonzero(~active).tolist(),
              "test": evaluate(prediction, data, test), "curves": curves,
              "train_selected": evaluate(predict(indices["train"]), data, indices["train"])}
    output.mkdir()
    torch.save(best_state, output / "weights.pt")
    np.savez(output / "preprocessing.npz", input_offset=input_norm.offset, input_scale=input_norm.scale,
             output_scale=output_norm.scale, asinh_scale=asinh_scale, active=active)
    np.savez(output / "test-predictions.npz", predicted=prediction, indices=test)
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets, config, provenance = load_run(args.output)
    print(json.dumps({"datasets": {name: audit["counts"] for name, (_, audit) in datasets.items()}, "variants": 12}), flush=True)
    if args.dry_run:
        return
    if torch.__version__ != "2.5.1+cpu":
        raise RuntimeError(f"Expected torch 2.5.1+cpu, found {torch.__version__}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output = args.output / "training"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    summary = {"status": "running", "source": revision(), "torch": torch.__version__,
               "config": config, "provenance": provenance, "datasets": {}, "variants": [], "failures": []}

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    save()
    try:
        for mechanism, (data, audit) in datasets.items():
            destination = output / mechanism
            destination.mkdir()
            np.savez(destination / "dataset.npz", **data)
            test = np.flatnonzero(data["split"] == "test")
            audit["zero_baseline"] = evaluate(np.zeros_like(data["delta"][test]), data, test)
            summary["datasets"][mechanism] = audit
            for target in config["targets"]:
                for precision in config["precisions"]:
                    try:
                        result = fit_variant(data, config, target, precision, destination / f"{target}-{precision}",
                                             started + config["wall_time_seconds"])
                        result["mechanism"] = mechanism
                        summary["variants"].append(result)
                        print(json.dumps({"mechanism": mechanism, "target": target, "precision": precision,
                                          "test_budget_p99": result["test"]["budget_p99"]}), flush=True)
                    except TimeoutError:
                        raise
                    except Exception as error:
                        summary["failures"].append({"mechanism": mechanism, "target": target,
                                                    "precision": precision, "error": str(error)})
                    save()
        summary["status"] = "complete" if len(summary["variants"]) == 12 else "incomplete"
    except TimeoutError:
        summary["status"] = "time_limit"
    finally:
        save()


if __name__ == "__main__":
    main()
