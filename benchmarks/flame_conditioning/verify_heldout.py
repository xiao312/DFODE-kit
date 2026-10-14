"""Independently reconcile saved offline test scores without fitting models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores


def expected_names(plan):
    names = ["zero-baseline"]
    for run in plan["models"]:
        directory = Path(run["directory"])
        summary = json.loads((directory / "summary.json").read_text())
        run_name = run.get("name", directory.name)
        variants = [row["name"] for row in summary["variants"]]
        names.extend(f"{run_name}--{name}" for name in variants)
        for name in variants:
            if name.endswith("-state-boxcox"):
                prefix = name.removesuffix("state-boxcox")
                if prefix + "signed-power" in variants:
                    names.append(f"{run_name}--{prefix}fixed-hybrid")
    historical = {(item["kind"], mode) for item in plan.get("historical", []) for mode in item["modes"]}
    names.extend(f"historical--{kind}--{mode}" for kind, mode in sorted(historical))
    for mode in ("source-formula", "stable-adapter"):
        if {("state-boxcox", mode), ("signed-power", mode)} <= historical:
            names.append(f"historical--fixed-hybrid--{mode}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate frozen model identity")
    return set(names)


def verify(test, evaluation):
    manifest = json.loads((test / "manifest.json").read_text())
    saved = json.loads((evaluation / "summary.json").read_text())
    if (manifest["status"] not in ("complete", "complete_with_exclusions")
            or saved["status"] != "complete"
            or saved["test_manifest_sha256"] != sha256(test / "manifest.json")):
        raise ValueError("Require complete evaluation of this exact test")
    for filename, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                          ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(test / filename) != manifest[key]:
            raise ValueError(f"Test hash mismatch: {filename}")
    plan = json.loads((test / "frozen-plan.json").read_text())
    for run in plan["models"] + plan.get("historical", []):
        for filename, digest in run["sha256"].items():
            if sha256(Path(run["directory"]) / filename) != digest:
                raise ValueError("Frozen model evidence changed")
    actual_names = [model["name"] for model in saved["models"]]
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != expected_names(plan):
        raise ValueError("Saved model list differs from the frozen comparison")
    source = np.load(test / "source-states.npz", allow_pickle=False)
    labels = np.load(test / "labels.npz", allow_pickle=False)
    accepted = labels["accepted"]
    states, delta = source["states"][accepted], labels["delta"][accepted]
    records = []
    for model in saved["models"]:
        path = evaluation / f'{model["name"]}.npz'
        with np.load(path, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["cell_ids"], source["sample_row"][accepted])
            if arrays["prediction"].shape != delta.shape or not np.isfinite(arrays["prediction"]).all():
                raise ValueError("Invalid saved test prediction")
            if model["name"] == "zero-baseline" and np.any(arrays["prediction"] != 0):
                raise ValueError("The zero baseline must predict zero increments")
            populations = {name for name in ("uniform", "balanced") if source[name][accepted].any()}
            if set(model["populations"]) != populations:
                raise ValueError("Test population missing or added")
            for population in sorted(populations):
                mask = source[population][accepted]
                if saved["sample_counts"][population] != {"selected": int(source[population].sum()), "accepted": int(mask.sum())}:
                    raise ValueError("Test population counts differ")
                actual = recompute(states[mask], arrays["prediction"][mask], delta[mask],
                                   test / "mechanism.yaml", manifest["interval_s"])
                assert_scores(actual, model["populations"][population])
                records.append({"name": model["name"], "population": population,
                                "prediction_sha256": sha256(path), **actual})
    return {"status": "verified", "source": source_revision(), "models": len(saved["models"]),
            "population_scores": len(records), "records": records,
            "evaluation_summary_sha256": sha256(evaluation / "summary.json"),
            "scope": "Independent main physical metrics on saved offline test predictions"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    result = verify(args.test, args.evaluation)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
