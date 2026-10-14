"""Compile small verified metadata, without Torch or access to private raw arrays."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from benchmarks.flame_conditioning.extract import sha256
from ..review import primary
from .checks import check_result, check_pair, matrix_names


def collect(root):
    status = json.loads((root / "campaign-status.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    names = matrix_names()
    if (status["status"] != "complete" or verification["status"] != "verified"
            or [r["name"] for r in verification["results"]] != names
            or [r["name"] for r in status["stages"]] != names
            or any(r["status"] != "verified" for r in status["stages"])
            or not all(verification.get(k) is True for k in
                ("paired_warmup", "common_inputs", "common_initial_weights", "historical_state_control_parity"))):
        raise ValueError("Require the complete verified 24-fit matrix")
    bound = {r["name"]:r["result_sha256"] for r in verification["results"]}
    results, files = {}, []
    queries = {key:[] for key in ("models", "history", "curves", "species", "bins")}
    for name in names:
        path = root / name
        result = json.loads((path / "result.json").read_text())
        check = json.loads((path / "verification.json").read_text())
        digest = sha256(path / "result.json")
        if digest != bound[name] or check["result_sha256"] != digest or check["status"] != "verified":
            raise ValueError("Result binding differs")
        if (not all(check.get(k) is True for k in ("exact_model_replay", "independent_paired_counts", "physical_checks", "frozen_50k_preprocessing"))
                or sha256(path / "environment.json") != result["artifacts"]["environment.json"]):
            raise ValueError("Missing checks or changed environment")
        check_result(result)
        config = result["config"]
        seed, target, objective = (config[k] for k in ("seed", "target", "objective"))
        if name != f"{seed}-{target}-{objective}":
            raise ValueError("Result identity differs")
        first_name = f"{seed}-state-boxcox-coordinate"
        if name != first_name:
            check_pair(results[first_name], result, False)
        elif result.get("exact_historical_control_parity") is not True:
            raise ValueError("Missing historical control parity")
        coordinate_name = f"{seed}-{target}-coordinate"
        if name != coordinate_name:
            check_pair(results[coordinate_name], result, True)
        results[name] = result
        for filename in ("result.json", "verification.json", "environment.json"):
            files.append(dict(name=f"{name}/{filename}", sha256=sha256(path / filename)))
        for policy in ("increment-reference-v1", "state-endpoint-v1"):
            identity = dict(recipe=f"matched200k-{target}-{objective}", target=target, objective=objective, seed=seed, policy=policy)
            train, dev = (primary(result[split][policy]) for split in ("training", "development"))
            physical = result["development_physical"]
            queries["models"].append(dict(identity, trainingRate=train["component_pass_fraction"],
                developmentRate=dev["component_pass_fraction"], stateRate=dev["state_pass_fraction"],
                p99=dev["normalized_quantiles"]["p99"], zeroRate=primary(result["zero_baseline"][policy])["component_pass_fraction"],
                negativeRate=physical["negative_endpoint_fraction"], correctionRate=physical["inverse_domain_correction_fraction"],
                fitSeconds=result["training_wall_seconds"], updates=result["updates_completed"], trainingCount=200000,
                inferenceMicroseconds=1e6*result["inference"]["median_wall_seconds"]/result["inference"]["states"], independentTest=None))
            for row in result["history"]:
                for split in ("training", "development"):
                    queries["history"].append(dict(identity, split=split, series=f"{objective}: {split}",
                        updates=row["updates"], seconds=row["elapsed_wall_seconds"], acceptanceRate=row[split][policy]["component_pass_fraction"]))
            for row in result["development"][policy]["grid"]:
                queries["curves"].append(dict(identity, atol=row["atol"], rtol=row["rtol"], role=row["role"],
                    acceptanceRate=row["component_pass_fraction"], stateRate=row["state_pass_fraction"]))
            for row in result["development"][policy]["per_species"]:
                queries["species"].append(dict(identity, species=row["species"], acceptanceRate=row["component_pass_fraction"], p99=row["normalized_quantiles"]["p99"]))
            for row in result["development"][policy]["magnitude_bins"]:
                queries["bins"].append(dict(identity, **row))
    for filename in ("verification.json", "campaign-status.json"):
        files.append(dict(name=filename, sha256=sha256(root / filename)))
    return queries, files, datetime.fromtimestamp(status["ended_unix"], timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    queries, files, cutoff = collect(args.campaign)
    source = dict(type="file", files=files, executedAt=cutoff,
        evidenceFlow=[dict(title="Verified factorial campaign", detail="Read all 24 result, verification and environment records. Require campaign-bound SHA256, frozen configs, exact work, common initialization and objective-paired warmup. Saved-array replay and independent counts/physical checks ran on the server. Extract sanitized summaries only; no test-set access.")],
        metricDefinitions=[dict(label="Frozen development baseline", componentIds=["matched-opening", "matched-method", "matched-models", "matched-history", "matched-tolerance", "matched-next"],
            definition="58 non-AR species, 1023 development states, 200k training rows per seed. Error is abs(predicted increment-reference increment). Primary allowance is 1e-15+.1*abs(reference increment), or reference endpoint for state policy. Whole-state pass requires all 58 components. p99 is error/allowance. Inference is median of five synchronized 1023-state GPU batches including preprocessing, transfers and FP64 inverse; not single-state latency. Fit wall includes diagnostics, excludes final verification and label generation.")],
        caveats=["Final checkpoints; repeated same-flame development use, not independent generalization.", "Subset label checks are not a precision certificate for every row.", "GBCT and power use common normalization/schedule, not original paper recipes."])
    for name, rows in queries.items():
        snapshot["queries"]["matched_targets_"+name] = dict(rows=rows, source=source)
    snapshot.update(generatedAt=datetime.now(timezone.utc).isoformat(), buildStatus="complete")
    args.output.write_text(json.dumps(snapshot,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({name:len(rows) for name,rows in queries.items()}))


if __name__ == "__main__":
    main()
