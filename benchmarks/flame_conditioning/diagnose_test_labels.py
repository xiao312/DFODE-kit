"""Diagnose strict test-label rejections without changing labels or scoring models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import endpoint, direct_increment
from benchmarks.flame_conditioning.extract import sha256, source_revision


def selected_failures(states, records, accepted):
    """Fixed, score-free selection: first, worst negativity, temperature quantiles."""
    failed = np.flatnonzero(~accepted)
    if not len(failed):
        raise ValueError("There are no rejected labels to diagnose")
    ordered = failed[np.argsort(states[failed, 0], kind="stable")]
    quantiles = ordered[np.linspace(0, len(ordered) - 1, min(6, len(ordered)), dtype=int)]
    worst = min(failed, key=lambda i: records[i]["diagnostics"]["minimum_mass_fraction"])
    return np.unique([failed[0], worst, *quantiles])


def rejection_profile(states, records, accepted, solver_atol):
    failed = np.flatnonzero(~accepted)
    diagnostics = [records[i]["diagnostics"] for i in failed]
    edges = [0, 305, 500, 1000, 1500, 2000, 3000]
    bins = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (states[:, 0] >= low) & (states[:, 0] < high)
        bins.append({"lower_K": low, "upper_K": high, "selected": int(mask.sum()),
                     "rejected": int((mask & ~accepted).sum())})
    minima = [entry["minimum_mass_fraction"] for entry in diagnostics]
    return {
        "selected": len(states), "accepted": int(accepted.sum()), "rejected": len(failed),
        "negative_endpoint_rows": sum(value < 0 for value in minima),
        "most_negative_endpoint": min(minima, default=0.0),
        "negative_magnitude_over_solver_atol_max": max((-value / solver_atol for value in minima), default=0.0),
        "other_constraint_failures": sum(
            abs(item["mass_delta_sum"]) > 1e-10 or item["element_delta_max"] > 1e-10
            or abs(item["temperature_change_K"]) > 1e-8
            or abs(item["relative_density_change"]) > 1e-10 for item in diagnostics),
        "temperature_bins": bins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; preserve the failed test and earlier diagnosis")
    manifest = json.loads((args.test / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_exclusions"):
        raise ValueError("Require completed label generation")
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(args.test / name) != manifest[key]:
            raise ValueError(f"Input hash mismatch: {name}")
    with np.load(args.test / "source-states.npz", allow_pickle=False) as arrays:
        states, cell_ids = arrays["states"], arrays["sample_row"]
    with np.load(args.test / "labels.npz", allow_pickle=False) as arrays:
        delta, accepted = arrays["delta"], arrays["accepted"]
    records = [json.loads(line) for line in (args.test / "label-records.jsonl").read_text().splitlines()]
    if (len(records) != len(states) or any(row["row"] != i or row["cell"] != cell_ids[i]
                                         for i, row in enumerate(records))):
        raise ValueError("Diagnostic rows do not match the selected cells")
    if not np.isfinite(delta).all() or any("diagnostics" not in row for row in records):
        raise ValueError("This diagnosis requires finite completed solves with endpoint diagnostics")
    mechanism = args.test / "mechanism.yaml"
    if ct.Solution(str(mechanism)).species_names != manifest["species_names"]:
        raise ValueError("Species order differs from the test manifest")
    selected = selected_failures(states, records, accepted)
    result = {"status": "planned", "source": source_revision(),
              "test_manifest_sha256": sha256(args.test / "manifest.json"),
              "label_records_sha256": sha256(args.test / "label-records.jsonl"),
              "profile": rejection_profile(states, records, accepted, manifest["cvode_atol"]),
              "selected_rows": selected.tolist(), "checks": [],
              "scope": "Reference diagnosis only. No model scoring, label clipping, or acceptance changes."}
    print(json.dumps({key: value for key, value in result.items() if key != "source"}), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    started = time.monotonic()
    result["status"] = "running"
    deadline = started + 600
    for index in selected:
        if time.monotonic() >= deadline:
            raise TimeoutError("Reference diagnosis reached its 600-second limit")
        row = states[index]
        state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
        checks = {
            "fresh": endpoint(mechanism, state, atol=1e-21),
            "tighter_absolute": endpoint(mechanism, state, atol=1e-24),
            "step_limited": endpoint(mechanism, state, atol=1e-21, max_step=1e-7),
            "direct": direct_increment(mechanism, state, rtol=1e-9, atol=1e-5, deadline=deadline),
            "direct_tight": direct_increment(mechanism, state, rtol=1e-11, atol=1e-7, deadline=deadline),
        }
        weight = 1e-12 + 1e-6 * np.abs(row[2:])
        differences = {name: float(np.max(np.abs(np.asarray(item["delta"]) - delta[index]) / weight))
                       for name, item in checks.items()}
        record = {"row": int(index), "cell": int(cell_ids[index]), "temperature_K": state["T"],
                  "saved": records[index], "saved_delta": delta[index].tolist(), "checks": checks,
                  "difference_from_saved_budget_max": differences}
        with (args.output / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        result["checks"].append({"row": int(index), "cell": int(cell_ids[index]),
                                 "temperature_K": state["T"], "differences": differences,
                                 "endpoint_minima": {name: item["diagnostics"]["minimum_mass_fraction"]
                                                     for name, item in checks.items()}})
        result["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result["checks"][-1]), flush=True)
    result["status"] = "complete"
    result["records_sha256"] = sha256(args.output / "records.jsonl")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
