"""Add verified offline evidence to the existing report without private paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def primary(summary):
    return next(row for row in summary["tolerances"] if row["atol"] == 1e-15 and row["rtol"] == .1)


def compile_rows(directories):
    models, curves, audited, species, magnitudes, files = [], [], [], [], [], []
    identity, zero, seeds = None, None, set()
    for directory in directories:
        evaluation_path = directory / "evaluation/summary.json"
        evaluation = read(evaluation_path)
        training_path = directory / "training/summary.json"
        training = read(training_path)
        checked = read(directory / "acceptance-verification.json")
        replay = read(directory / "replay.json")
        if (evaluation["status"] != "complete" or training["status"] != "complete"
                or checked["status"] != "verified" or replay["status"] != "verified"
                or not replay["training_metrics_checked"]
                or checked["evaluation_summary_sha256"] != sha256(evaluation_path)
                or replay["training_summary_sha256"] != sha256(training_path)
                or evaluation["training_summary_sha256"] != sha256(training_path)):
            raise ValueError("Require complete hash-bound evaluation and replay checks")
        config = training["plan"]["config"]
        if (config["updates"] != 2000 or config["checkpoint_selection"] != "final"
                or config["training_sizes"] != [2000, 10000]
                or config["hidden_widths"] != [800]*4 or config["batch_size"] != 256
                or config["precisions"] != ["float32"] or config["loss"] != "l1"
                or config["activation"] != "gelu"):
            raise ValueError("First-stage comparison recipe differs")
        seed = config["seed"]
        if seed in seeds or seed != evaluation["plan"]["seed"]:
            raise ValueError("Repeated or mismatched seed")
        seeds.add(seed)
        current = (evaluation["dataset_manifest_sha256"], evaluation["audit_summary_sha256"])
        if identity is not None and current != identity:
            raise ValueError("The seeds must share the exact dataset and audit")
        identity = current
        if zero is not None and zero != evaluation["zero_baseline"]:
            raise ValueError("The zero baseline changed between seeds")
        zero = evaluation["zero_baseline"]
        pairs = {(row["target"], row["training_count"]) for row in evaluation["models"]}
        if len(evaluation["models"]) != 8 or pairs != {(target, size) for target in config["targets"] for size in (2000, 10000)}:
            raise ValueError("Require all four targets at both accepted training sizes")
        for row in evaluation["models"]:
            key = {"target": row["target"], "seed": seed, "trainingCount": row["training_count"]}
            p = primary(row["nominal"])
            physical = row["physical"]
            models.append({**key, "states": p["states"], "components": p["components"],
                "componentPassCount": p["component_pass_count"], "statePassCount": p["state_pass_count"],
                "componentRate": p["component_pass_fraction"], "stateRate": p["state_pass_fraction"],
                "sspiRate": row["nominal"]["sspi"]["value"],
                "trainingSeconds": row["training_process_seconds"], "wallSeconds": row["training_wall_seconds"],
                "inferenceSeconds": row["inference"]["median_process_seconds"],
                "negativeComponentRate": physical["negative_endpoint_fraction"],
                "massDriftP99": physical["mass_increment_drift"]["p99"],
                "inverseCorrectionRate": physical["inverse_domain_correction_fraction"]})
            for item in row["nominal"]["tolerances"]:
                curves.append({**key, "atol": item["atol"], "rtol": item["rtol"],
                    "relativeLabel": f'{100*item["rtol"]:g}%',
                    "componentRate": item["component_pass_fraction"], "stateRate": item["state_pass_fraction"]})
            audited.append({**key, **primary(row["audited_subset"])})
            species.extend({**key, **item} for item in row["nominal"]["per_species"])
            magnitudes.extend({**key, **item} for item in row["nominal"]["magnitude_bins"])
        for relative in ("evaluation/summary.json", "training/summary.json", "acceptance-verification.json", "replay.json"):
            files.append({"name": f"seed-{seed}/{relative}", "sha256": sha256(directory/relative)})
    if seeds != {20261011, 20261012}:
        raise ValueError("Retain both predeclared seeds")
    for item in zero["nominal"]["tolerances"]:
        curves.append({"target": "zero", "seed": None, "trainingCount": 0, "atol": item["atol"], "rtol": item["rtol"],
            "relativeLabel": f'{100*item["rtol"]:g}%', "componentRate": item["component_pass_fraction"], "stateRate": item["state_pass_fraction"]})
    metadata = {**evaluation["dataset"], **evaluation["plan"], "datasetHash": identity[0], "auditHash": identity[1],
                "zeroComponentRate": primary(zero["nominal"])["component_pass_fraction"],
                "zeroStateRate": primary(zero["nominal"])["state_pass_fraction"],
                "zeroSspiRate": zero["nominal"]["sspi"]["value"], "totalModels": len(models)}
    return {"offline_models": models, "offline_curves": curves, "offline_audit": audited,
            "offline_species": species, "offline_magnitudes": magnitudes, "offline_contract": [metadata]}, files


def extend_snapshot(snapshot, rows, files):
    now = datetime.now(timezone.utc).isoformat()
    for key, values in rows.items():
        snapshot["queries"][key] = {"rows": values, "source": {
            "provider": "DFODE-kit verified offline experiments", "name": key, "files": files,
            "executedAt": now,
            "metricDefinitions": [{"label": "Increment acceptance", "definition":
                "abs(prediction-reference) <= atol + rtol*abs(reference). Components exclude AR; a state passes all remaining species. Primary atol=1e-15, rtol=0.1. Full-population results are nominal; reference qualification applies only to 16 audited evaluation states.",
                "componentIds": ["offline-opening", "offline-contract", "offline-tolerance", "offline-cost", "offline-all-models", "offline-magnitudes", "offline-audit", "offline-next"]},
                {"label": "Training CPU cost", "definition": "Process CPU seconds recorded by the training process, including input preparation and periodic evaluations. Excludes label generation and independent verification.", "componentIds": ["offline-cost", "offline-all-models"]},
                {"label": "Inference cost", "definition": "Median process seconds from five full-batch predictions after warmup. Displayed milliseconds per state equals 1000 times that median divided by the evaluated state count. Not single-state latency.", "componentIds": ["offline-cost", "offline-all-models"]},
                {"label": "SSPI", "definition": "Among components with absolute reference increment below 1e-15, the fraction whose absolute prediction is also below 1e-15. Zero change scores 100%; this alone does not show useful accuracy.", "componentIds": ["offline-all-models", "offline-next"]}],
            "caveats": ["Two held-out snapshots of one flame, not an independent case or CFD transfer test.",
                        "Process CPU seconds exclude label generation and verification. Two jobs each had a half-CPU quota; wall time includes throttling.",
                        "Both seeds remain separate. Target scales are fit again at each nested training size."],
            "evidenceFlow": [{"title": "Verify artifact identity", "detail": "Require successful saved-model replay and independent tolerance-count checks, bound to summary SHA256 hashes."},
                             {"title": "Retain the full comparison", "detail": "Keep four targets, two sizes and both fixed seeds. Add the zero-change baseline. Preserve all old report queries."}]}}
    snapshot["generatedAt"] = now
    snapshot["buildStatus"] = "updating"
    snapshot.setdefault("report", {})["asOf"] = "2026-10-09"
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("seeds", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    rows, files = compile_rows(args.seeds)
    args.output.write_text(json.dumps(extend_snapshot(read(args.snapshot), rows, files), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"models": len(rows["offline_models"]), "curves": len(rows["offline_curves"]), "output_sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
