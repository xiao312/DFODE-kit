"""Independently check a temperature-stratified subset of augmented labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import direct_increment, endpoint
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.scout import summarize_record, temperature_selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-per-split", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new audit directory")
    if not 2 <= args.count_per_split <= 64:
        parser.error("Choose 2 to 64 states per split")
    data, physics, manifest = load_dataset(args.dataset)
    selected = {split: temperature_selection(rows["states"], args.count_per_split) for split, rows in data.items()}
    plan = {split: [int(data[split]["source_indices"][i]) for i in indices] for split, indices in selected.items()}
    print(json.dumps({"rows": plan, "interval_s": manifest["config"]["interval_s"]}), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + 900
    mechanism = args.dataset / "mechanism.yaml"
    interval = manifest["config"]["interval_s"]
    summary = {"status": "running", "source": source_revision(),
               "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
               "plan": plan, "records": [], "failures": [],
               "scope": "Selected augmented states only; not certification of every label"}

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    save()
    for split, indices in selected.items():
        for index in indices:
            row = data[split]["states"][index]
            state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
            record = {"split": split, "source_row": int(data[split]["source_indices"][index]),
                      "state": state, "checks": {"tight": {"delta": data[split]["delta"][index].tolist()}}, "failures": []}
            for name, rtol, atol, max_step in [("study-tolerance", 1e-6, 1e-10, None),
                                               ("tighter", 1e-12, 1e-21, None),
                                               ("step-limited", 1e-12, 1e-21, interval / 10)]:
                try:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Augmented audit deadline")
                    record["checks"][name] = endpoint(mechanism, state, interval, rtol, atol, max_step)
                except Exception as error:
                    record["failures"].append({"check": name, "error": str(error)})
            for name, rtol, atol in [("direct", 1e-9, 1e-5), ("direct-tighter", 1e-11, 1e-7)]:
                try:
                    record["checks"][name] = direct_increment(mechanism, state, interval, rtol, atol, deadline)
                except Exception as error:
                    record["failures"].append({"check": name, "error": str(error)})
            with (args.output / "records.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            summary["records"].append({"split": split, "source_row": record["source_row"],
                                       "temperature_K": state["T"], **summarize_record(record)})
            summary["failures"].extend(record["failures"])
            print(json.dumps({"split": split, "source_row": record["source_row"],
                              "uncertainty_budget_max": summary["records"][-1].get("uncertainty_budget_max")}), flush=True)
            save()
            if time.monotonic() >= deadline:
                summary["status"] = "time_limit"
                save()
                return
    checked = [record for record in summary["records"] if record["status"] == "checked"]
    passed = len(checked) == 2 * args.count_per_split and all(all(record["budget_fit"]) for record in checked)
    summary.update(status="complete" if not summary["failures"] else "incomplete", reference_subset_pass=passed,
                   components_checked=len(checked) * len(physics["species_names"]))
    if checked:
        summary["budget_fit_fraction"] = float(np.mean([record["budget_fit"] for record in checked]))
        summary["relative_fit_fraction"] = float(np.mean([record["relative_fit"] for record in checked]))
    save()
    if not passed:
        raise SystemExit("Reference subset failed; inspect evidence before training")


if __name__ == "__main__":
    main()
