"""Score a frozen model list on an audited reserved snapshot, without tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cantera as ct
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify import load_predictor
from benchmarks.flame_conditioning.historical import load_historical


def hybrid_prediction(states, boxcox, power):
    """Fixed thresholds from the paper, never inferred from this test set."""
    prediction = np.zeros_like(states[:, 2:])
    correction = np.zeros_like(prediction, dtype=bool)
    for selected, predict in (((states[:, 0] >= 305) & (states[:, 0] < 1000), power),
                              (states[:, 0] >= 1000, boxcox)):
        if selected.any():
            prediction[selected], correction[selected] = predict(states[selected])
    return prediction, correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    torch.set_num_threads(1)
    manifest = json.loads((args.test / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_exclusions"):
        raise ValueError("Reserved labels are incomplete")
    for filename, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                          ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(args.test / filename) != manifest[key]:
            raise ValueError(f"Test provenance mismatch: {filename}")
    audit = json.loads((args.audit / "summary.json").read_text())
    if (audit["status"] != "complete" or audit.get("budget_fit_fraction") != 1.0
            or audit["source_manifest"]["states_sha256"] != manifest["states_sha256"]):
        raise ValueError("Require a passing independent reference audit on this snapshot")
    source = np.load(args.test / "source-states.npz", allow_pickle=False)
    labels = np.load(args.test / "labels.npz", allow_pickle=False)
    accepted = labels["accepted"]
    if accepted.mean() < .9:
        raise ValueError("More than 10% excluded test labels; repair references first")
    # The scout recomputes references independently; compare its tight endpoint
    # with the actual labels used below, not just with other scout solvers.
    for line in (args.audit / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        index = record["source_row"]
        if accepted[index]:
            difference = np.abs(labels["delta"][index] - record["checks"]["tighter"]["delta"])
            if np.any(difference > .01 * (1e-12 + 1e-6 * source["states"][index, 2:])):
                raise ValueError("Stored test label disagrees with independent fresh endpoint")
    states, delta = source["states"][accepted], labels["delta"][accepted]
    gas = ct.Solution(str(args.test / "mechanism.yaml"))
    if gas.species_names != manifest["species_names"]:
        raise ValueError("Test species order mismatch")
    gas.TP = 298.15, ct.one_atm
    physics = {"species_names": gas.species_names, "element_matrix": element_matrix(gas),
               "formation_enthalpies": gas.standard_enthalpies_RT * ct.gas_constant * gas.T / gas.molecular_weights,
               "molecular_weights": gas.molecular_weights, "interval": manifest["interval_s"]}
    masks = {name: source[name][accepted] for name in ("uniform", "balanced")}
    plan = json.loads((args.test / "frozen-plan.json").read_text())
    # Verify every frozen byte before loading any model or writing metrics.
    for run in plan["models"] + plan.get("historical", []):
        for relative, digest in run["sha256"].items():
            if sha256(Path(run["directory"]) / relative) != digest:
                raise ValueError(f"Frozen model artifact changed: {relative}")
    if args.dry_run:
        print(json.dumps({"training_runs": len(plan["models"]), "accepted_cells": len(states),
                          "populations": {name: int(mask.sum()) for name, mask in masks.items()}, "output": str(args.output)}))
        return
    args.output.mkdir(parents=True)
    result = {"source": source_revision(), "test_manifest_sha256": sha256(args.test / "manifest.json"),
              "audit_summary_sha256": sha256(args.audit / "summary.json"), "models": [],
              "scope": "Offline single-snapshot test; no coupled CFD or post-test tuning claim",
              "sample_counts": {name: {"selected": int(source[name].sum()), "accepted": int(mask.sum())}
                                for name, mask in masks.items()}}

    def score(name, predicted, corrected, diagnostics=None):
        entry = {"name": name, "populations": {}, "diagnostics": diagnostics or {}}
        for population, mask in masks.items():
            if mask.any():
                entry["populations"][population] = physical_scores(predicted[mask], delta[mask], states[mask], corrected[mask], **physics)
        result["models"].append(entry)
        np.savez_compressed(args.output / f"{name}.npz", prediction=predicted, correction=corrected,
                            cell_ids=source["sample_row"][accepted])
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({"name": name, "uniform_budget_p99": entry["populations"]["uniform"]["budget_error"]["p99"]}), flush=True)

    score("zero-baseline", np.zeros_like(delta), np.zeros_like(delta, dtype=bool))
    for run in plan["models"]:
        directory = Path(run["directory"])
        training = json.loads((directory / "summary.json").read_text())
        config = training["plan"]["config"]
        predictors = {}
        for variant in training["variants"]:
            predict, _ = load_predictor(directory / variant["name"], config)
            predictors[variant["name"]] = predict
            score(f'{directory.name}--{variant["name"]}', *predict(states))
        for name, boxcox in predictors.items():
            if name.endswith("-state-boxcox"):
                prefix = name.removesuffix("state-boxcox")
                power = predictors.get(prefix + "signed-power")
                if power is not None:
                    score(f"{directory.name}--{prefix}fixed-hybrid", *hybrid_prediction(states, boxcox, power))
    historical_predictors = {}
    for control in plan.get("historical", []):
        for mode in control["modes"]:
            predict, _ = load_historical(control["directory"], gas.species_names, mode)
            historical_predictors[(control["kind"], mode)] = predict
            predicted, corrected = predict(states)
            score(f'historical--{control["kind"]}--{mode}', predicted, corrected, dict(predict.diagnostics))
    for mode in ("source-formula", "stable-adapter"):
        boxcox = historical_predictors.get(("state-boxcox", mode))
        power = historical_predictors.get(("signed-power", mode))
        if boxcox is not None and power is not None:
            score(f"historical--fixed-hybrid--{mode}", *hybrid_prediction(states, boxcox, power),
                  diagnostics={"historical_training_overlap_not_excluded": True,
                               "reconstruction": mode, "thresholds_K": [305, 1000]})
    result["status"] = "complete"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
