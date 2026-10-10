"""Replay frozen offline models and measure acceptance versus tolerance and cost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.verify import load_predictor
from benchmarks.offline_accuracy.metrics import summarize


def audit_subset(validation, audit):
    by_source = {int(row): index for index, row in enumerate(validation["source_indices"])}
    rows = [row for row in audit["records"] if row["split"] == "validation"]
    if not rows or any(row["status"] != "checked" for row in rows):
        raise ValueError("Require complete checked evaluation audit rows")
    indices = np.array([by_source[row["source_row"]] for row in rows])
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Duplicate audit rows")
    reference = np.array([row["reference_delta"] for row in rows])
    uncertainty = np.array([row["uncertainty_estimate"] for row in rows])
    # Endpoint spacing also limits confidence when comparing an endpoint-derived label.
    initial = validation["states"][indices, 2:]
    spacing = np.maximum(np.abs(np.spacing(initial)), np.abs(np.spacing(initial + reference)))
    return indices, reference, np.maximum(uncertainty, spacing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, manifest = load_dataset(args.dataset)
    training = json.loads((args.training / "summary.json").read_text())
    audit_path = args.audit / "summary.json"
    audit = json.loads(audit_path.read_text())
    config = training["plan"]["config"]
    if (training["status"] != "complete" or config.get("checkpoint_selection") != "final"
            or training["dataset_manifest_sha256"] != sha256(args.dataset / "manifest.json")
            or training["audit_summary_sha256"] != sha256(audit_path)
            or audit["dataset_manifest_sha256"] != training["dataset_manifest_sha256"]
            or not audit["reference_subset_pass"] or audit["status"] != "complete"
            or config.get("pressure_bounds_Pa") != manifest["config"].get("pressure_bounds_Pa")):
        raise ValueError("Require matching complete offline data, audited labels, and final-checkpoint models")
    expected = len(config["targets"]) * len(config["precisions"]) * len(config["training_sizes"])
    if len(training["variants"]) != expected:
        raise ValueError("Incomplete matched comparison")
    for variant in training["variants"]:
        if variant["selected_step"] != config["updates"] or variant["updates_completed"] != config["updates"]:
            raise ValueError("Every model must use the fixed final update")
    validation = data["validation"]
    indices, references, uncertainty = audit_subset(validation, audit)
    plan = {"seed": config["seed"], "models": expected, "evaluation_states": len(validation["states"]),
            "audited_evaluation_states": len(indices), "pressure_bounds_Pa": config["pressure_bounds_Pa"],
            "primary_atol": 1e-15, "primary_rtol": .1, "inference_repeats": 5,
            "scope": "Offline held-out snapshots from one flame realization; no CFD transfer test"}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir()
    result = {"status": "running", "source": source_revision(), "plan": plan,
              "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
              "training_summary_sha256": sha256(args.training / "summary.json"),
              "audit_summary_sha256": sha256(audit_path), "models": [],
              "dataset": {"accepted": len(data["train"]["states"]),
                          "label_generation_wall_seconds": manifest["elapsed_seconds"],
                          "excluded": {split: value["labels_completed"]-value["labels_accepted"]
                                       for split, value in manifest["splits"].items()}},
              "runtime": {"torch": torch.__version__, "threads": 1, "device": "cpu"}}

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))

    for variant in training["variants"]:
        directory = args.training / variant["name"]
        predict, _ = load_predictor(directory, config)
        predicted, corrections = predict(validation["states"])
        with np.load(directory / "validation-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["source_indices"], validation["source_indices"])
            np.testing.assert_allclose(predicted, saved["prediction"], rtol=1e-13, atol=1e-30)
            np.testing.assert_array_equal(corrections, saved["correction"])
        durations = []
        for _ in range(5):
            wall, process = time.perf_counter(), time.process_time()
            predict(validation["states"])
            durations.append({"wall_seconds": time.perf_counter()-wall, "process_seconds": time.process_time()-process})
        item = {"name": variant["name"], "target": variant["target"], "seed": config["seed"],
                "training_count": variant["training_count"], "updates": variant["updates_completed"],
                "training_wall_seconds": variant["elapsed_seconds"], "training_process_seconds": variant["process_seconds"],
                "inference": {"states": len(validation["states"]), "repeats": durations,
                              "median_wall_seconds": float(np.median([row["wall_seconds"] for row in durations])),
                              "median_process_seconds": float(np.median([row["process_seconds"] for row in durations]))},
                "physical": variant["validation"],
                "nominal": summarize(predicted, validation["delta"], physics["species_names"]),
                "audited_subset": summarize(predicted[indices], references, physics["species_names"], uncertainty),
                "artifact_sha256": {name: sha256(directory / name) for name in
                    ("weights.pt", "preprocessing.npz", "validation-predictions.npz", "training-indices.npy")}}
        result["models"].append(item)
        save()
    result["zero_baseline"] = {"nominal": summarize(np.zeros_like(validation["delta"]), validation["delta"], physics["species_names"]),
        "audited_subset": summarize(np.zeros_like(references), references, physics["species_names"], uncertainty)}
    result["status"] = "complete"
    save()
    print(json.dumps({"status": "complete", "models": len(result["models"])}))


if __name__ == "__main__":
    main()
