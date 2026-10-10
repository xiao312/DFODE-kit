"""Bounded fixed-temperature/volume reference audit on saved flame states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cantera as ct
import numpy as np
import scipy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import direct_increment, endpoint
from benchmarks.flame_conditioning.extract import sha256, source_revision


def temperature_selection(states, count):
    """Select distinct rows distributed over temperature, without labels."""
    if count < 2 or count > len(states):
        raise ValueError("Scout size must be between two and the number of source states")
    order = np.argsort(states[:, 0], kind="stable")
    selected = []
    available = np.ones(len(states), dtype=bool)
    for target in np.linspace(states[:, 0].min(), states[:, 0].max(), count):
        candidates = order[available[order]]
        index = int(candidates[np.argmin(np.abs(states[candidates, 0] - target))])
        selected.append(index)
        available[index] = False
    return np.array(selected, dtype=int)


def summarize_record(record):
    state = record["state"]
    checks = record["checks"]
    required = ["tight", "tighter", "step-limited", "direct", "direct-tighter"]
    if any(name not in checks for name in required + ["study-tolerance"]):
        return {"status": "missing_check"}
    deltas = np.array([checks[name]["delta"] for name in required])
    reference = np.asarray(checks["direct-tighter"]["delta"])
    uncertainty = np.max(np.abs(deltas - reference), axis=0)
    initial = np.asarray(state["Y"])
    budget = 1e-12 + 1e-6 * np.abs(initial)
    spacing = np.maximum(np.abs(np.spacing(initial)), np.abs(np.spacing(initial + reference)))
    relative_fit = np.abs(reference) > 100 * np.maximum(uncertainty, spacing)
    return {
        "status": "checked", "budget_fit": (uncertainty <= .01 * budget).tolist(),
        "relative_fit": relative_fit.tolist(), "reference_delta": reference.tolist(),
        "uncertainty_estimate": uncertainty.tolist(),
        "uncertainty_budget_max": float(np.max(uncertainty / budget)),
        "zero_reference_count": int(np.sum(reference == 0)),
        "endpoint_lost_nonzero_count": int(np.sum((np.asarray(checks["tighter"]["delta"]) == 0) & relative_fit)),
        "original_tolerance_budget_max": float(np.max(np.abs(np.asarray(checks["study-tolerance"]["delta"]) - reference) / budget)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    if not 1 <= args.wall_seconds <= 3600:
        parser.error("Wall limit must be between 1 and 3600 seconds")
    data = np.load(args.source / "source-states.npz", allow_pickle=False)
    manifest = json.loads((args.source / "manifest.json").read_text())
    if sha256(args.source / "source-states.npz") != manifest["states_sha256"]:
        raise ValueError("Source states checksum mismatch")
    mechanism = args.source / "mechanism.yaml"
    if sha256(mechanism) != manifest["mechanism_sha256"]:
        raise ValueError("Mechanism checksum mismatch")
    indices = temperature_selection(data["states"], args.count)
    print(json.dumps({"rows": indices.tolist(), "count": len(indices), "interval_s": 1e-6,
                      "constraint": "fixed temperature and volume", "wall_seconds": args.wall_seconds}), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    summary = {
        "status": "running", "source": source_revision(), "source_manifest": manifest,
        "versions": {"cantera": ct.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "interval_s": 1e-6, "constraint": "fixed temperature and volume", "count_planned": len(indices),
        "count_completed": 0, "failures": [], "records": [],
    }

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    save()
    for index in indices:
        if time.monotonic() >= deadline:
            summary["status"] = "time_limit"
            break
        row = data["states"][index]
        state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
        record = {"source_row": int(index), "snapshot": str(data["snapshot"][index]),
                  "sample_row": int(data["sample_row"][index]), "state": state, "checks": {}, "failures": []}
        settings = [("study-tolerance", 1e-6, 1e-10, None), ("default", 1e-9, 1e-15, None),
                    ("tight", 1e-12, 1e-18, None), ("tighter", 1e-12, 1e-21, None),
                    ("step-limited", 1e-12, 1e-21, 1e-7)]
        for name, rtol, atol, max_step in settings:
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Scout deadline")
                record["checks"][name] = endpoint(mechanism, state, rtol=rtol, atol=atol, max_step=max_step)
            except Exception as error:
                record["failures"].append({"check": name, "error": str(error)})
        for name, rtol, atol in [("direct", 1e-9, 1e-5), ("direct-tighter", 1e-11, 1e-7)]:
            try:
                record["checks"][name] = direct_increment(mechanism, state, rtol=rtol, atol=atol, deadline=deadline)
            except Exception as error:
                record["failures"].append({"check": name, "error": str(error)})
        # Preserve each completed/failed interval before summary analysis.
        with (args.output / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        scores = summarize_record(record)
        summary["records"].append({"source_row": int(index), "temperature_K": state["T"], **scores})
        summary["count_completed"] += 1
        summary["failures"].extend({"source_row": int(index), **failure} for failure in record["failures"])
        print(json.dumps({"completed": summary["count_completed"], "temperature_K": state["T"],
                          "status": scores["status"], "uncertainty_budget_max": scores.get("uncertainty_budget_max")}), flush=True)
        save()
    if summary["status"] == "running":
        summary["status"] = "complete" if not summary["failures"] else "incomplete"
    checked = [record for record in summary["records"] if record["status"] == "checked"]
    if checked:
        summary["components_checked"] = sum(len(record["budget_fit"]) for record in checked)
        summary["budget_fit_fraction"] = float(np.mean([record["budget_fit"] for record in checked]))
        summary["relative_fit_fraction"] = float(np.mean([record["relative_fit"] for record in checked]))
    save()


if __name__ == "__main__":
    main()
