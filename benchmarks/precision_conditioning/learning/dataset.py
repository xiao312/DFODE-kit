"""Validate reference rows and construct whole-parent learning splits."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from analysis import assess
from targets import validate_group_splits


def partition(records, manifest, experiment, mechanism):
    """Reject uncertain intervals as whole rows; never silently impute labels."""
    metadata = next(item for item in manifest["mechanisms"] if item["id"] == mechanism)
    assignments = {f"{mechanism}-{float(temperature):g}K": split
                   for temperature, split in experiment["split_by_temperature"].items()}
    rows, excluded, seen = [], [], set()
    for record in records:
        if record["mechanism"] != mechanism:
            continue
        if record["id"] in seen:
            raise ValueError("Duplicate interval ID")
        seen.add(record["id"])
        parent = record["trajectory_id"]
        if parent not in assignments:
            raise ValueError("Unknown parent trajectory")
        checked = assess(record, manifest["config"])
        if record["failures"] or checked is None:
            excluded.append({"id": record["id"], "parent": parent, "split": assignments[parent], "reason": "missing_or_failed_solver"})
            continue
        if not np.all(checked["budget_fit"]):
            excluded.append({"id": record["id"], "parent": parent, "split": assignments[parent],
                             "reason": "reference_disagreement", "components": np.flatnonzero(~checked["budget_fit"]).tolist()})
            continue
        initial = np.r_[record["state"]["T"], record["state"]["Y"]]
        features = np.r_[initial[0], record["state"]["P"], initial[1:], np.log10(record["interval_s"])]
        if initial.size != metadata["species_count"] + 1 or not np.all(np.isfinite(features)):
            raise ValueError("Invalid state shape or non-finite input")
        if not all(np.all(np.isfinite(checked[key])) for key in ("reference", "weights", "uncertainty")):
            raise ValueError("Non-finite reference data")
        rows.append({"id": record["id"], "parent": parent, "split": assignments[parent],
                     "inputs": features, "initial": initial, "delta": checked["reference"],
                     "weights": checked["weights"], "relative_mask": checked["relative_fit"]})
    if not rows:
        raise ValueError("No eligible rows")
    validate_group_splits([r["parent"] for r in rows], [r["split"] for r in rows])
    counts = {label: sum(row["split"] == label for row in rows) for label in ("train", "validation", "test")}
    for label, minimum in experiment["minimum_accepted"].items():
        if counts[label] < minimum:
            raise ValueError(f"Insufficient {label} rows: {counts[label]} < {minimum}")
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    # Count exact feature overlap across splits. Fail rather than hiding a leakage risk.
    feature_splits = {}
    for values, label in zip(arrays["inputs"], arrays["split"]):
        key = values.tobytes()
        if key in feature_splits and feature_splits[key] != label:
            raise ValueError("Exact input duplicate across splits")
        feature_splits[key] = label
    audit = {"mechanism": mechanism, "source_rows": len(seen), "accepted_rows": len(rows),
             "counts": counts, "excluded": excluded, "species": metadata["species"],
             "parents": assignments, "exact_cross_split_duplicates": 0}
    return arrays, audit


def load_run(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    experiment = json.loads((output / "experiment.json").read_text())
    if manifest["status"] != "complete" or manifest["trajectory_failures"]:
        raise ValueError("Require a complete dataset with no parent failures")
    records = [json.loads(line) for line in (output / "intervals.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != manifest["planned_intervals"]:
        raise ValueError("Record count does not match manifest")
    datasets = {item["id"]: partition(records, manifest, experiment, item["id"]) for item in manifest["mechanisms"]}
    return datasets, experiment, {
        "pilot_source": manifest["source"],
        "intervals_sha256": hashlib.sha256((output / "intervals.jsonl").read_bytes()).hexdigest(),
        "mechanisms": manifest["mechanisms"], "versions": manifest["versions"],
        "data_seconds": manifest["elapsed_seconds"], "config": manifest["config"],
    }
