"""Add verified data-size and throughput evidence without replacing earlier queries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256
from .review import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new review output")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    models, files, sizes, identities = [], [], [], {}
    for root in args.campaign:
        first = json.loads((root / "seed-20261011/fuel-state/result.json").read_text())
        count = first["training_count"]
        if count not in (10000, 50000, 200000) or count in sizes:
            raise ValueError("Each declared size must appear once")
        rows, _, artifacts, identity = collect(root, count)
        models.extend(rows)
        sizes.append(count)
        identities[count] = identity
        files.extend(dict(name=f"n{count}/{item['name']}", sha256=item["sha256"]) for item in artifacts)
        if count != 10000:
            for path in root.glob("seed-*/fuel-*/result.json"):
                result = json.loads(path.read_text())
                if result["hashes"]["comparison_dataset_manifest"] != snapshot["queries"]["offline_contract"]["rows"][0]["datasetHash"]:
                    raise ValueError("Expanded run has a different original dataset")
    if 10000 not in sizes or 50000 not in sizes:
        raise ValueError("Include the original 10k and complete 50k comparisons")
    original = snapshot["queries"]["offline_contract"]["rows"][0]
    if identities[10000] != (original["datasetHash"], original["auditHash"]):
        raise ValueError("Original report identity differs")
    throughput = json.loads(args.throughput.read_text())
    if throughput["status"] != "verified" or throughput["dataset_manifest_sha256"] != identities[50000][0]:
        raise ValueError("Throughput did not pass on the checked 50k dataset")
    measurements = throughput["measurements"]
    if (sorted((r["workers"], r["repetition"]) for r in measurements) !=
            [(1, 0), (1, 1), (4, 0), (4, 1), (8, 0), (8, 1)] or
            not all(r["exact_serial_labels"] and r["exact_acceptance_flags"] for r in measurements)):
        raise ValueError("Incomplete parallel parity benchmark")
    now = datetime.now(timezone.utc).isoformat()
    sources = dict(type="file", executedAt=now, files=files,
        evidenceFlow=[dict(title="Saved experiment evidence", detail="Load every result and verification hash for both Fuel recipes and both seeds at each size. Check fixed schedules, update counts, row presentations and unchanged original dataset. No independent test is opened.")],
        metricDefinitions=[dict(label="Acceptance and work", componentIds=["fuel-scale-opening", "fuel-scale-results", "fuel-scale-terms"],
            definition="58 non-argon species, 1023 unchanged development states. Primary allowance is 1e-15 + 0.1*abs(reference increment), or 1e-15 + 0.1*abs(reference endpoint) for the separately selected state policy. Whole-state pass requires every component to pass. Fit time excludes references and final verification. Row presentations include repeated rows, not unique examples.")],
        caveats=["Larger datasets also receive more optimizer updates at fixed epochs; this is not an isolated data-only effect.",
                 "The same four training and two development snapshots remain correlated parts of one flame.",
                 "Nominal label metrics and an audited subset do not certify all labels or solver-level accuracy."])
    snapshot["queries"]["fuel_scale_models"] = dict(rows=sorted(models, key=lambda r:(r["trainingCount"],r["recipe"],r["seed"],r["policy"])), source=sources)
    snapshot["queries"]["fuel_label_throughput"] = dict(rows=measurements, source=dict(type="file", executedAt=now,
        files=[dict(name=args.throughput.name, sha256=sha256(args.throughput))],
        evidenceFlow=[dict(title="Identical-row worker benchmark", detail="Relabel 2048 evenly spaced saved training rows with 1, 4, 8 workers, then reverse the order. Require exact increments and acceptance flags. Timing includes worker startup and transport, not full-dataset augmentation or final export.")],
        metricDefinitions=[dict(label="Reference throughput", componentIds=["fuel-throughput", "fuel-throughput-note"],
            definition="Rows divided by measured elapsed seconds for CPU Cantera/CVODE labeling. Each process has one numerical thread. This is not GPU training or a CFD speedup.")]))
    snapshot.update(generatedAt=now, buildStatus="complete")
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(sizes=sorted(sizes), models=len(models), worker_measurements=len(measurements))))


if __name__ == "__main__":
    main()
