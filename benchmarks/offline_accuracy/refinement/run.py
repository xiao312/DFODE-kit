"""Validate and execute one immutable member of the refinement comparison."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify import load_predictor
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.metrics import summarize
from .fit import fit, prediction
from .plan import configuration, variants


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def inputs(dataset, audit_directory, base_directory, seed):
    data, physics, manifest = load_dataset(dataset)
    audit = json.loads((audit_directory / "summary.json").read_text())
    original = json.loads((base_directory / "summary.json").read_text())
    digest = sha256(dataset / "manifest.json")
    if (audit.get("status") != "complete" or not audit.get("reference_subset_pass")
            or audit.get("dataset_manifest_sha256") != digest
            or original.get("status") != "complete" or original.get("dataset_manifest_sha256") != digest
            or original.get("audit_summary_sha256") != sha256(audit_directory / "summary.json")):
        raise ValueError("Dataset, completed base run and passing audit must match")
    config = original["plan"]["config"]
    if (config["seed"] != seed or config["updates"] != 2000
            or config["checkpoint_selection"] != "final" or config["hidden_widths"] != [800]*4
            or manifest["config"].get("pressure_bounds_Pa") != [96258.75, 106391.25]):
        raise ValueError("Require the frozen original 2000-update offline experiment")
    for rows in data.values():
        if np.any((rows["states"][:, 1] < 96258.75) | (rows["states"][:, 1] > 106391.25)):
            raise ValueError("Pressure outside the frozen domain")
    selected = np.random.default_rng(seed).permutation(len(data["train"]["states"]))[:10000]
    if len(selected) != 10000:
        raise ValueError("This comparison requires exactly 10000 accepted training rows")
    training = {key: value[selected] for key, value in data["train"].items()}
    base_path = base_directory / "n10000-float32-state-boxcox"
    base, base_result = load_predictor(base_path, config)
    if base_result["status"] != "complete" or base_result["selected_step"] != 2000:
        raise ValueError("Require the original final base checkpoint")
    np.testing.assert_array_equal(np.load(base_path / "training-indices.npy"), training["source_indices"])
    hashes = {"dataset_manifest": digest, "audit_summary": sha256(audit_directory / "summary.json"),
              "base_summary": sha256(base_directory / "summary.json"),
              "base_weights": sha256(base_path / "weights.pt"),
              "base_preprocessing": sha256(base_path / "preprocessing.npz")}
    return training, data["validation"], physics, audit, base, base_result, hashes


def evaluate(model, preprocessing, training, validation, physics, audit, config, base, destination):
    predict = lambda states: prediction(model, preprocessing, states, config["target"], base)
    result = {}
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrections = predict(rows["states"])
        result[name] = summarize(predicted, rows["delta"], physics["species_names"])
        result[f"{name}_physical"] = physical_scores(predicted, rows["delta"], rows["states"], corrections, **physics)
        np.savez_compressed(destination / f"{name}-predictions.npz", prediction=predicted,
                            correction=corrections, source_indices=rows["source_indices"])
    selected, reference, uncertainty = audit_subset(validation, audit)
    predicted, _ = predict(validation["states"])
    result["audited_subset"] = summarize(predicted[selected], reference, physics["species_names"], uncertainty)
    result["zero_baseline"] = summarize(np.zeros_like(validation["delta"]), validation["delta"], physics["species_names"])
    durations = []
    for _ in range(5):
        wall, cpu = time.perf_counter(), time.process_time()
        predict(validation["states"])
        durations.append(dict(wall_seconds=time.perf_counter()-wall, process_seconds=time.process_time()-cpu))
    result["inference"] = dict(states=len(validation["states"]), repeats=durations,
        median_process_seconds=float(np.median([r["process_seconds"] for r in durations])),
        median_wall_seconds=float(np.median([r["wall_seconds"] for r in durations])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=[r["name"] for r in variants()])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.seed not in (20261011, 20261012):
        parser.error("Use one of the two frozen seeds")
    if torch.__version__ != "2.5.1+cpu":
        raise ValueError("Require the existing pinned Torch 2.5.1+cpu image")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training, validation, physics, audit, base, base_result, hashes = inputs(args.dataset, args.audit, args.base, args.seed)
    print(json.dumps(dict(variants=variants(), hashes=hashes, seed=args.seed)), flush=True)
    if args.dry_run:
        return
    if args.variant is None:
        parser.error("Select one --variant per bounded execution")
    config = configuration(args.variant, args.seed)
    args.output.mkdir(parents=True)
    result = dict(status="running", config=config, source=source_revision(), hashes=hashes,
                  runtime=dict(torch=torch.__version__, threads=1, dtype="float32", reconstruction="float64"))
    save(args.output / "result.json", result)
    try:
        frozen = base if config["residual"] else None
        model, preprocessing, fitted = fit(training, validation, physics["species_names"], config, args.output, frozen)
        result.update(fitted)
        result.update(evaluate(model, preprocessing, training, validation, physics, audit, config, frozen, args.output))
        result["base_training_process_seconds"] = base_result["process_seconds"] if frozen else 0.
        result["total_training_process_seconds"] = result["training_process_seconds"] + result["base_training_process_seconds"]
        result["total_parameter_count"] = result["parameter_count"] + (sum(p.numel() for p in model.parameters()) if frozen else 0)
        result["artifacts"] = {name: sha256(args.output / name) for name in (
            "weights.pt", "optimizer.pt", "preprocessing.npz", "training-indices.npy",
            "training-predictions.npz", "validation-predictions.npz")}
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise
    save(args.output / "result.json", result)
    print(json.dumps(dict(status="complete", name=args.variant, process_seconds=result["total_training_process_seconds"])))


if __name__ == "__main__":
    main()
