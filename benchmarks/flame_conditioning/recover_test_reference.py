"""Explicitly recover sign-noise reference rows; preserve all raw signed labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import endpoint, direct_increment
from benchmarks.flame_conditioning.extract import sha256, source_revision


def constraints_pass(check, negative_floor):
    """Allow only declared sub-tolerance sign noise; never clip a value."""
    values = [check[key] for key in ("minimum_mass_fraction", "mass_delta_sum", "element_delta_max",
                                    "temperature_change_K", "relative_density_change")]
    return bool(np.isfinite(values).all() and check["minimum_mass_fraction"] >= -negative_floor
                and abs(check["mass_delta_sum"]) <= 1e-10 and check["element_delta_max"] <= 1e-10
                and abs(check["temperature_change_K"]) <= 1e-8
                and abs(check["relative_density_change"]) <= 1e-10)


def recovery_decision(initial, saved_delta, saved_diagnostics, checks, negative_floor=1e-21):
    """All checks must agree with the unchanged label within the existing 1% budget."""
    required = {"fresh", "tighter_absolute", "direct", "direct_tight"}
    if set(checks) != required:
        raise ValueError("Every independent recovery check is required")
    saved_delta = np.asarray(saved_delta)
    if not np.isfinite(saved_delta).all() or saved_diagnostics["minimum_mass_fraction"] >= 0:
        return False, None
    if not constraints_pass(saved_diagnostics, negative_floor):
        return False, None
    weight = 1e-12 + 1e-6 * np.abs(initial)
    differences = []
    for result in checks.values():
        delta = np.asarray(result["delta"])
        if delta.shape != saved_delta.shape or not np.isfinite(delta).all():
            return False, None
        if not constraints_pass(result["diagnostics"], negative_floor):
            return False, None
        differences.append(float(np.max(np.abs(delta - saved_delta) / weight)))
    uncertainty = max(differences)
    return uncertainty <= .01, uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new; never replace the original strict-reference run")
    original = json.loads((args.test / "manifest.json").read_text())
    if original["status"] != "complete_with_exclusions" or original.get("reference_recovery"):
        raise ValueError("Require an original completed strict test with exclusions")
    if original["cvode_atol"] != 1e-21 or original["cvode_rtol"] != 1e-12 or original["interval_s"] != 1e-6:
        raise ValueError("This amendment applies only to the recorded fixed-T/V reference recipe")
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(args.test / name) != original[key]:
            raise ValueError(f"Original reference hash mismatch: {name}")
    with np.load(args.test / "source-states.npz", allow_pickle=False) as arrays:
        states, cell_ids = arrays["states"], arrays["sample_row"]
    with np.load(args.test / "labels.npz", allow_pickle=False) as arrays:
        delta, strict = arrays["delta"], arrays["accepted"]
    if not np.isfinite(delta).all():
        raise ValueError("This recovery cannot replace missing or nonfinite reference increments")
    records = [json.loads(line) for line in (args.test / "label-records.jsonl").read_text().splitlines()]
    if len(records) != len(states) or any(row["row"] != i or row["cell"] != cell_ids[i]
                                        for i, row in enumerate(records)):
        raise ValueError("Reference diagnostic records do not match source cells")
    mechanism = args.test / "mechanism.yaml"
    if ct.Solution(str(mechanism)).species_names != original["species_names"]:
        raise ValueError("Reference species order mismatch")
    failed = np.flatnonzero(~strict)
    plan = {"policy": "raw-sign-noise-with-per-row-independent-checks-v1",
            "original_test_manifest_sha256": sha256(args.test / "manifest.json"),
            "original_labels_sha256": original["labels_sha256"],
            "negative_endpoint_floor": 1e-21, "maximum_disagreement_budget": .01,
            "required_checks": ["fresh", "tighter_absolute", "direct", "direct_tight"],
            "original_selected": len(states), "original_accepted": int(strict.sum()),
            "rows_to_check": len(failed), "wall_seconds": 1800,
            "disclosure": "Post-reference, pre-model-score amendment. No clipping, refitting, or test resampling."}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    (args.output / "recovery-plan.json").write_text(json.dumps(plan, indent=2))
    for name in ("source-states.npz", "mechanism.yaml", "frozen-plan.json", "label-records.jsonl"):
        shutil.copyfile(args.test / name, args.output / name)
    accepted, recovered = strict.copy(), np.zeros_like(strict)
    uncertainty = np.full(len(states), np.nan)
    manifest = {**original, "status": "recovery_running", "source": source_revision(),
                "reference_recovery": plan, "recovery_rows_completed": 0}
    started = time.monotonic()
    deadline = started + 1800
    summary = {"status": "running", "source": source_revision(), "plan": plan,
               "completed": 0, "recovered": 0, "still_excluded": len(failed)}

    def save():
        np.savez_compressed(args.output / "labels.npz", delta=delta, accepted=accepted)
        np.savez_compressed(args.output / "reference-quality.npz", strict_accepted=strict,
                            recovered=recovered, uncertainty_budget=uncertainty)
        manifest.update(labels_accepted=int(accepted.sum()), labels_sha256=sha256(args.output / "labels.npz"),
                        reference_quality_sha256=sha256(args.output / "reference-quality.npz"),
                        recovery_elapsed_seconds=time.monotonic() - started)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        summary.update(completed=manifest["recovery_rows_completed"], recovered=int(recovered.sum()),
                       still_excluded=int((~accepted).sum()), elapsed_seconds=time.monotonic() - started)
        (args.output / "recovery-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    save()
    for index in failed:
        if time.monotonic() >= deadline:
            manifest["status"] = summary["status"] = "time_limit"
            save()
            return
        row = states[index]
        state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
        record = {"row": int(index), "cell": int(cell_ids[index]), "recovered": False}
        try:
            checks = {
                "fresh": endpoint(mechanism, state, atol=1e-21),
                "tighter_absolute": endpoint(mechanism, state, atol=1e-24),
                "direct": direct_increment(mechanism, state, rtol=1e-9, atol=1e-5, deadline=deadline),
                "direct_tight": direct_increment(mechanism, state, rtol=1e-11, atol=1e-7, deadline=deadline),
            }
            passed, difference = recovery_decision(row[2:], delta[index], records[index]["diagnostics"], checks)
            record.update(checks=checks, recovered=passed, uncertainty_budget_max=difference)
            if difference is not None:
                uncertainty[index] = difference
            recovered[index] = accepted[index] = passed
        except Exception as error:
            record["error"] = str(error)
        with (args.output / "recovery-records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        manifest["recovery_rows_completed"] += 1
        if manifest["recovery_rows_completed"] % 50 == 0:
            save()
            print(json.dumps(summary), flush=True)
    # Keep the original failure records in original_strict_failures, not as new failures.
    manifest["original_strict_failures"] = original["failures"]
    manifest["failures"] = [item for item in original["failures"] if not accepted[item["row"]]]
    manifest["status"] = "complete" if accepted.all() else "complete_with_exclusions"
    manifest["recovery_records_sha256"] = sha256(args.output / "recovery-records.jsonl")
    summary["status"] = "complete"
    summary["uncertainty_budget_max"] = float(np.nanmax(uncertainty)) if np.isfinite(uncertainty).any() else None
    save()
    summary["test_manifest_sha256"] = sha256(args.output / "manifest.json")
    (args.output / "recovery-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
