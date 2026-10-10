"""Pressure-only post-score diagnosis with fresh counterfactual chemistry labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cantera as ct
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import endpoint, direct_increment, element_matrix
from benchmarks.flame_conditioning.evaluate_heldout import hybrid_prediction
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.recover_test_reference import constraints_pass
from benchmarks.flame_conditioning.verify import load_predictor
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores


def pressure_copy(states, pressure):
    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError("Counterfactual pressure must be positive and finite")
    copied = np.asarray(states).copy()
    copied[:, 1] = pressure
    return copied


def checked_label(checks, initial):
    if set(checks) != {"cvode", "step_limited", "direct", "direct_tight"}:
        raise ValueError("All pressure-reference checks are required")
    reference = np.asarray(checks["cvode"]["delta"])
    if any(np.asarray(item["delta"]).shape != np.asarray(initial).shape for item in checks.values()):
        raise ValueError("Pressure-reference increments must match the species vector")
    weight = 1e-12 + 1e-6 * np.abs(initial)
    uncertainty = max(float(np.max(np.abs(np.asarray(item["delta"]) - reference) / weight))
                      for item in checks.values())
    passed = all(np.isfinite(item["delta"]).all() and constraints_pass(item["diagnostics"], 1e-21)
                 for item in checks.values())
    return bool(passed and uncertainty <= .01), uncertainty, reference


def heat_source(states, delta, physics):
    rho = states[:, 1] / (ct.gas_constant * states[:, 0] *
                         np.sum(states[:, 2:] / physics["molecular_weights"], axis=1))
    return -(delta @ physics["formation_enthalpies"]) * rho / physics["interval"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; the frozen test must remain unchanged")
    torch.set_num_threads(1)
    manifest = json.loads((args.test / "manifest.json").read_text())
    audit = json.loads((args.audit / "summary.json").read_text())
    run = json.loads((args.training / "summary.json").read_text())
    if (manifest["status"] not in ("complete", "complete_with_exclusions") or run["status"] != "complete"
            or audit["status"] != "complete" or audit["budget_fit_fraction"] != 1
            or audit["source_manifest"]["states_sha256"] != manifest["states_sha256"]):
        raise ValueError("Require the completed frozen test, passing audit, and completed models")
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(args.test / name) != manifest[key]:
            raise ValueError("Frozen test identity mismatch")
    frozen = json.loads((args.test / "frozen-plan.json").read_text())
    matches = [entry for entry in frozen["models"] if Path(entry["directory"]) == args.training.resolve()]
    if len(matches) != 1:
        raise ValueError("Primary run must be in the original frozen list")
    for name, digest in matches[0]["sha256"].items():
        if sha256(args.training / name) != digest:
            raise ValueError("Frozen model artifact changed")
    targets = {row["target"] for row in run["variants"]}
    if len(run["variants"]) != 4 or targets != {"state-boxcox", "signed-power", "budget-linear", "scaled-asinh"}:
        raise ValueError("Require the complete primary four-target comparison")
    indices = np.asarray([row["source_row"] for row in audit["records"]], dtype=int)
    if len(indices) != 32 or len(set(indices.tolist())) != 32:
        raise ValueError("Use the fixed 32-state audit subset")
    with np.load(args.test / "source-states.npz", allow_pickle=False) as arrays:
        states, cells = arrays["states"][indices], arrays["sample_row"][indices]
    with np.load(args.test / "labels.npz", allow_pickle=False) as arrays:
        if not arrays["accepted"][indices].all():
            raise ValueError("Audit subset contains an excluded original label")
        original_delta = arrays["delta"][indices]
    pressures = []
    for variant in run["variants"]:
        with np.load(args.training / variant["name"] / "preprocessing.npz", allow_pickle=False) as arrays:
            pressures.append(float(arrays["x_offset"][1]))
    if len(set(pressures)) != 1:
        raise ValueError("The four frozen models do not share one training pressure mean")
    changed = pressure_copy(states, pressures[0])
    plan = {"rows": indices.tolist(), "cells": cells.tolist(), "pressure_Pa": pressures[0],
            "count": len(indices), "wall_seconds": 300,
            "test_manifest_sha256": sha256(args.test / "manifest.json"),
            "audit_summary_sha256": sha256(args.audit / "summary.json"),
            "training_summary_sha256": sha256(args.training / "summary.json"),
            "scope": "Post-score pressure-only diagnosis on the fixed audit subset; not new validation or a repair."}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    started = time.monotonic()
    deadline = started + 300
    mechanism = args.test / "mechanism.yaml"
    changed_delta = np.full_like(original_delta, np.nan)
    uncertainty_values = []
    summary = {"status": "reference_running", "source": source_revision(), "plan": plan, "models": [],
               "versions": {"cantera": ct.__version__, "numpy": np.__version__, "torch": torch.__version__}}
    for i, row in enumerate(changed):
        if time.monotonic() >= deadline:
            raise TimeoutError("Pressure diagnosis reached its 300-second limit")
        state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
        checks = {"cvode": endpoint(mechanism, state, atol=1e-21),
                  "step_limited": endpoint(mechanism, state, atol=1e-21, max_step=1e-7),
                  "direct": direct_increment(mechanism, state, rtol=1e-9, atol=1e-5, deadline=deadline),
                  "direct_tight": direct_increment(mechanism, state, rtol=1e-11, atol=1e-7, deadline=deadline)}
        passed, uncertainty, changed_delta[i] = checked_label(checks, row[2:])
        record = {"row": int(indices[i]), "cell": int(cells[i]), "state": state,
                  "checks": checks, "reference_pass": passed, "uncertainty_budget_max": uncertainty}
        with (args.output / "reference-records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if not passed:
            summary.update(status="reference_failed", failed_row=int(indices[i]))
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            raise ValueError("Counterfactual reference failed; do not change pressure or score models")
        uncertainty_values.append(uncertainty)
    np.savez_compressed(args.output / "paired-reference.npz", original_states=states,
                        changed_states=changed, original_delta=original_delta, changed_delta=changed_delta,
                        source_rows=indices, cell_ids=cells)
    gas = ct.Solution(str(mechanism))
    if gas.species_names != manifest["species_names"]:
        raise ValueError("Pressure diagnostic species order mismatch")
    gas.TP = 298.15, ct.one_atm
    physics = {"species_names": gas.species_names, "element_matrix": element_matrix(gas),
               "formation_enthalpies": gas.standard_enthalpies_RT * ct.gas_constant * gas.T / gas.molecular_weights,
               "molecular_weights": gas.molecular_weights, "interval": 1e-6}
    original_heat = heat_source(states, original_delta, physics)
    changed_heat = heat_source(changed, changed_delta, physics)
    rms = lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))
    baseline_rms = rms(original_heat)
    if baseline_rms == 0:
        raise ValueError("Zero reference heat RMS cannot support a relative response")
    summary.update(status="scoring", uncertainty_budget_max=max(uncertainty_values),
                   reference_heat_response_relative_rms=rms(changed_heat - original_heat) / baseline_rms,
                   paired_reference_sha256=sha256(args.output / "paired-reference.npz"))
    combined = np.concatenate([states, changed])
    predictors = {row["target"]: load_predictor(args.training / row["name"], run["plan"]["config"])[0]
                  for row in run["variants"]}
    predictors["fixed-hybrid"] = lambda rows: hybrid_prediction(rows, predictors["state-boxcox"], predictors["signed-power"])
    predictors["zero-baseline"] = lambda rows: (np.zeros_like(rows[:, 2:]), np.zeros_like(rows[:, 2:], dtype=bool))
    for name, predict in predictors.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("Pressure diagnosis reached its 300-second limit")
        prediction, correction = predict(combined)
        entry = {"target": name, "conditions": {}}
        for condition, selection, reference in (("original", slice(0, len(states)), original_delta),
                                                 ("training_mean_pressure", slice(len(states), None), changed_delta)):
            scores = physical_scores(prediction[selection], reference, combined[selection], correction[selection], **physics)
            independent = recompute(combined[selection], prediction[selection], reference, mechanism, 1e-6)
            assert_scores(independent, scores)
            entry["conditions"][condition] = scores
        entry["prediction_heat_response_relative_rms"] = rms(
            heat_source(changed, prediction[len(states):], physics) -
            heat_source(states, prediction[:len(states)], physics)) / baseline_rms
        path = args.output / f"{name}-paired-predictions.npz"
        np.savez_compressed(path, prediction=prediction, correction=correction)
        entry["predictions_sha256"] = sha256(path)
        summary["models"].append(entry)
    summary.update(status="complete", independent_physical_metrics_verified=True,
                   elapsed_seconds=time.monotonic() - started)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps({"status": summary["status"], "models": len(summary["models"]),
                      "seconds": summary["elapsed_seconds"], "reference_heat_response_relative_rms": summary["reference_heat_response_relative_rms"]}))


if __name__ == "__main__":
    main()
