"""Independently count offline acceptance and reconcile saved source identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256


def check_row(predicted, reference, saved, uncertainty=None):
    normalized = np.abs(predicted-reference) / (saved["atol"] + saved["rtol"]*np.abs(reference))
    passes = normalized <= 1
    expected = {"components": reference.size, "states": len(reference),
                "component_pass_count": np.count_nonzero(passes),
                "state_pass_count": sum(all(row) for row in passes)}
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f"Independent tolerance count differs: {key}")
    np.testing.assert_allclose(saved["component_pass_fraction"], expected["component_pass_count"]/reference.size)
    np.testing.assert_allclose(saved["state_pass_fraction"], expected["state_pass_count"]/len(reference))
    np.testing.assert_allclose(saved["normalized_error_p99"], np.percentile(normalized, 99), rtol=1e-12)
    np.testing.assert_allclose(saved["normalized_error_max"], normalized.max(), rtol=1e-12)
    if uncertainty is not None:
        allowance = saved["atol"] + saved["rtol"] * abs(reference)
        known = np.isfinite(uncertainty) & (uncertainty <= allowance / 10)
        passed = known & ((abs(predicted-reference) + uncertainty) / allowance <= 1)
        qualified = int(np.count_nonzero(known))
        known_states = int(sum(all(row) for row in known))
        success = int(np.count_nonzero(passed))
        success_states = int(sum(all(row) for row in passed))
        for key, value in {"qualified_components": qualified, "unknown_components": reference.size-qualified,
                           "qualified_states": known_states, "unknown_states": len(reference)-known_states,
                           "qualified_pass_count": success, "qualified_state_pass_count": success_states}.items():
            if saved[key] != value:
                raise ValueError(f"Independent uncertainty count differs: {key}")
        for key, numerator, denominator in (("qualified_pass_fraction", success, qualified),
                                             ("qualified_state_pass_fraction", success_states, known_states)):
            if denominator:
                np.testing.assert_allclose(saved[key], numerator/denominator)
            elif saved[key] is not None:
                raise ValueError("Unknown reference population must not report a pass fraction")


def check_summary(predicted, reference, saved, names, uncertainty=None):
    active = [i for i, name in enumerate(names) if name != "AR"]
    predicted, reference = predicted[:, active], reference[:, active]
    uncertainty = None if uncertainty is None else uncertainty[:, active]
    expected_pairs = {(a, r) for a in (1e-12, 1e-15, 1e-18) for r in (1., .1, .01, .001)}
    if len(saved["tolerances"]) != 12 or {(r["atol"], r["rtol"]) for r in saved["tolerances"]} != expected_pairs:
        raise ValueError("Frozen tolerance grid differs")
    for row in saved["tolerances"]:
        check_row(predicted, reference, row, uncertainty)
    if len(saved["per_species"]) != len(active):
        raise ValueError("Missing species results")
    for index, row in enumerate(saved["per_species"]):
        if row["species"] != names[active[index]]:
            raise ValueError("Species order differs")
        check_row(predicted[:, index:index+1], reference[:, index:index+1], row,
                  None if uncertainty is None else uncertainty[:, index:index+1])
    small = abs(reference) < 1e-15
    correct = small & (abs(predicted) < 1e-15)
    if saved["sspi"]["small_components"] != small.sum() or saved["sspi"]["small_predictions"] != correct.sum():
        raise ValueError("SSPI counts differ")
    if small.sum():
        np.testing.assert_allclose(saved["sspi"]["value"], correct.sum()/small.sum())
    elif saved["sspi"]["value"] is not None:
        raise ValueError("An empty SSPI population is unknown")
    edges = [0., 1e-30, 1e-20, 1e-15, 1e-12, 1e-9, 1e-6, None]
    if [(row["lower"], row["upper"]) for row in saved["magnitude_bins"]] != list(zip(edges[:-1], edges[1:])):
        raise ValueError("Magnitude bins differ")
    for row in saved["magnitude_bins"]:
        upper = np.inf if row["upper"] is None else row["upper"]
        mask = (abs(reference) >= row["lower"]) & (abs(reference) < upper)
        errors = abs(predicted-reference)[mask]
        count = int(np.count_nonzero(errors <= 1e-15 + .1*abs(reference[mask])))
        if row["components"] != mask.sum() or row["pass_count"] != count:
            raise ValueError("Magnitude-bin counts differ")
        if mask.sum():
            np.testing.assert_allclose(row["pass_fraction"], count/mask.sum())
            np.testing.assert_allclose(row["absolute_error_p99"], np.percentile(errors, 99), rtol=1e-12)
        elif row["pass_fraction"] is not None or row["absolute_error_p99"] is not None:
            raise ValueError("Empty bins must remain null")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    saved = json.loads((args.evaluation / "summary.json").read_text())
    training = json.loads((args.training / "summary.json").read_text())
    for key, path in (("dataset_manifest_sha256", args.dataset/"manifest.json"),
                      ("training_summary_sha256", args.training/"summary.json"),
                      ("audit_summary_sha256", args.audit/"summary.json")):
        if saved[key] != sha256(path):
            raise ValueError("Evaluation source hash mismatch")
    if saved["status"] != "complete" or {row["name"] for row in saved["models"]} != {row["name"] for row in training["variants"]}:
        raise ValueError("Evaluation model set differs from the completed run")
    data, physics, _ = load_dataset(args.dataset)
    validation = data["validation"]
    audit = json.loads((args.audit/"summary.json").read_text())
    records = [row for row in audit["records"] if row["split"] == "validation"]
    selected = [int(np.flatnonzero(validation["source_indices"] == row["source_row"])[0]) for row in records]
    references = np.array([row["reference_delta"] for row in records])
    initial = validation["states"][selected, 2:]
    uncertainty = np.maximum(np.array([row["uncertainty_estimate"] for row in records]),
        np.maximum(abs(np.spacing(initial)), abs(np.spacing(initial+references))))
    for model in saved["models"]:
        original = next(row for row in training["variants"] if row["name"] == model["name"])
        for output_key, input_key in (("target", "target"), ("training_count", "training_count"),
                                       ("updates", "updates_completed"), ("training_wall_seconds", "elapsed_seconds"),
                                       ("training_process_seconds", "process_seconds")):
            if model[output_key] != original[input_key]:
                raise ValueError("Model identity or training cost differs")
        inference = model["inference"]
        if inference["states"] != len(validation["states"]) or len(inference["repeats"]) != 5:
            raise ValueError("Inference timing population differs")
        for metric in ("wall_seconds", "process_seconds"):
            durations = [row[metric] for row in inference["repeats"]]
            if not np.isfinite(durations).all() or min(durations) <= 0:
                raise ValueError("Invalid inference duration")
            np.testing.assert_allclose(inference[f"median_{metric}"], np.median(durations))
        directory = args.training/model["name"]
        for name, digest in model["artifact_sha256"].items():
            if sha256(directory/name) != digest:
                raise ValueError("Saved model artifact changed")
        with np.load(directory/"validation-predictions.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["source_indices"], validation["source_indices"])
            predicted = arrays["prediction"]
        check_summary(predicted, validation["delta"], model["nominal"], physics["species_names"])
        check_summary(predicted[selected], references, model["audited_subset"], physics["species_names"], uncertainty)
    check_summary(np.zeros_like(validation["delta"]), validation["delta"], saved["zero_baseline"]["nominal"], physics["species_names"])
    check_summary(np.zeros_like(references), references, saved["zero_baseline"]["audited_subset"], physics["species_names"], uncertainty)
    result = {"status": "verified", "models": len(saved["models"]), "tolerance_pairs": 12,
              "per_species_magnitude_and_sspi_checked": True,
              "evaluation_summary_sha256": sha256(args.evaluation/"summary.json"),
              "scope": "Independent saved-metric reconciliation, not a bound on all reference errors"}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
