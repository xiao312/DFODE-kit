"""Publish checked small summaries; never load weights or open independent tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256
from ..review import primary
from .plan import configuration, update_policy


POLICIES = ("increment-reference-v1", "state-endpoint-v1")


def validate_result(result, check, result_hash, environment_hash, recipe, seed, count):
    config = configuration(recipe, seed)
    expected_schedule = []
    last_rate = None
    for completed in range(config["updates"]):
        rate, reset = update_policy(config, completed)
        if rate != last_rate:
            expected_schedule.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
            last_rate = rate
    if (result["status"] != "complete" or check["status"] != "verified"
            or check["result_sha256"] != result_hash
            or result["artifacts"]["environment.json"] != environment_hash
            or result["config"] != config or result["training_count"] != count
            or result["development_count"] != 1023 or result["independent_test_count"] != 0
            or result["updates_completed"] != 6000 or result["row_presentations"] != 60000000
            or result["effective_batch_size"] != 10000 or result["parameter_count"] != 2018458
            or result["completed_pool_passes"] != 60000000//count
            or result["schedule"] != expected_schedule
            or [r["updates"] for r in result["history"]] != list(range(1000, 6001, 1000))
            or not all(check.get(k) for k in ("exact_model_replay", "independent_paired_counts",
                                             "physical_checks", "frozen_50k_preprocessing"))):
        raise ValueError("Matched-work result failed configuration, work or verification checks")


def pair_rows(models):
    pairs = []
    for recipe in ("fuel-state-matched-work", "fuel-power-matched-work"):
        for seed in (20261011, 20261012):
            for policy in POLICIES:
                selected = [r for r in models if (r["recipe"], r["seed"], r["policy"]) == (recipe, seed, policy)]
                if sorted(r["trainingCount"] for r in selected) != [50000, 200000]:
                    raise ValueError("Each comparison needs exactly one 50k and one 200k result")
                small, large = sorted(selected, key=lambda r:r["trainingCount"])
                pairs.append(dict(recipe=recipe, seed=seed, policy=policy,
                    smallRate=small["developmentRate"], largeRate=large["developmentRate"],
                    changePoints=100*(large["developmentRate"]-small["developmentRate"])))
    return pairs


def collect(root):
    campaign_path = root / "verification.json"
    campaign = json.loads(campaign_path.read_text())
    if campaign["status"] != "verified" or not all(campaign.get(k) for k in
            ("equal_work", "common_normalization", "nested_selections", "identical_initial_weights")):
        raise ValueError("Require complete remote campaign verification")
    bound = {r["name"]: r["result_sha256"] for r in campaign["results"]}
    if len(bound) != 8 or len(campaign["results"]) != 8:
        raise ValueError("Campaign must bind exactly eight results")
    models, history, files, identities, initializations = [], [], [], set(), {}
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            paired = []
            for count in (50000, 200000):
                directory = root / f"{seed}-{recipe}-n{count}"
                result_hash = sha256(directory / "result.json")
                if bound.get(directory.name) != result_hash:
                    raise ValueError("Result differs from remote campaign verification")
                result = json.loads((directory / "result.json").read_text())
                check = json.loads((directory / "verification.json").read_text())
                runtime = json.loads((directory / "environment.json").read_text())
                validate_result(result, check, result_hash, sha256(directory / "environment.json"), recipe, seed, count)
                paired.append(result)
                identity = result["initial_weights_sha256"]
                if seed in initializations and initializations[seed] != identity:
                    raise ValueError("Seed initialization differs")
                initializations[seed] = identity
                identities.add((result["hashes"]["dataset_manifest"], result["hashes"]["audit_summary"],
                                result["hashes"]["comparison_dataset_manifest"]))
                display_id = recipe+"-matched-work"
                for policy in POLICIES:
                    train, dev = primary(result["training"][policy]), primary(result["development"][policy])
                    models.append(dict(recipe=display_id, seed=seed, policy=policy, trainingCount=count,
                        trainingRate=train["component_pass_fraction"], developmentRate=dev["component_pass_fraction"],
                        stateRate=dev["state_pass_fraction"], p99=dev["normalized_quantiles"]["p99"],
                        updates=result["updates_completed"], presentations=result["row_presentations"],
                        fitSeconds=result["training_wall_seconds"],
                        negativeRate=result["development_physical"]["negative_endpoint_fraction"],
                        correctionRate=result["development_physical"]["inverse_domain_correction_fraction"],
                        device=runtime["device"], independentTest=None))
                    for record in result["history"]:
                        if record["row_presentations"] != record["updates"]*10000:
                            raise ValueError("History exposure differs")
                        for split in ("training", "development"):
                            history.append(dict(recipe=display_id, seed=seed, policy=policy, trainingCount=count,
                                split=split, series=f"{count//1000}k {split}", updates=record["updates"],
                                acceptanceRate=record[split][policy]["component_pass_fraction"],
                                seconds=record["elapsed_wall_seconds"]))
                for name in ("result.json", "verification.json", "environment.json"):
                    files.append(dict(name=f"{directory.name}/{name}", sha256=sha256(directory / name)))
            for key in ("hashes", "preprocessing_array_sha256", "initial_weights_sha256"):
                if paired[0][key] != paired[1][key]:
                    raise ValueError("Paired data, scaler or initialization differs")
    if len(identities) != 1:
        raise ValueError("Dataset identity differs")
    files.append(dict(name="verification.json", sha256=sha256(campaign_path)))
    return models, history, pair_rows(models), files, identities.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    models, history, pairs, files, identity = collect(args.campaign)
    if identity[2] != snapshot["queries"]["offline_contract"]["rows"][0]["datasetHash"]:
        raise ValueError("Original report dataset differs")
    # Bind this control to the completed 200k reference identity, not just a row count.
    declared = snapshot["queries"]["fuel_scale_models"]["source"]["files"]
    source_hashes = {r["name"]:r["sha256"] for r in declared}
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            result = json.loads((args.campaign / f"{seed}-{recipe}-n50000/result.json").read_text())
            for size in (50000, 200000):
                if result["hashes"][f"source_{size//1000}k_result"] != source_hashes.get(f"n{size}/seed-{seed}/{recipe}/result.json"):
                    raise ValueError("Prior scale evidence differs from matched-work source")
    now = datetime.now(timezone.utc).isoformat()
    source = dict(type="file", executedAt=now, files=files,
        evidenceFlow=[dict(title="Eight verified GPU fits", detail="Remote verification replays saved weights, counts passes independently, checks physical metrics and frozen 50k preprocessing. Campaign verification binds all eight result hashes, nested IDs, common development rows and initial weights. This compiler rechecks the small summaries and configuration; it does not replay weights locally.")],
        metricDefinitions=[dict(label="Matched-work comparison", componentIds=["matched-opening", "matched-protocol", "matched-pairs", "matched-models", "matched-history", "matched-next"],
            definition="Every fit: 6000 updates, batches of 10000, 60M presentations. Primary a=1e-15,r=0.1; allowance scales with abs(reference increment) or abs(reference endpoint), as selected. Same signed-increment error; 58 species and 1023 development states. Pair change is 100*(200k rate - 50k rate) percentage points. Final checkpoint, both seeds; independent test unopened. GPU fit wall includes full-pool diagnostics, excludes final verification and reference generation.")],
        caveats=["Within-recipe data-size comparison; recipes retain different coordinate normalization and rate schedules.",
                 "Common scalers use 50k training rows; this is not the original fixed-epoch source recipe.",
                 "Same-flame development states are not independent flame cases. Reference subset audit is not per-label certification.",
                 "No 1M expansion or solver-level accuracy claim follows automatically."])
    for name, rows in (("models", models), ("history", history), ("pairs", pairs)):
        snapshot["queries"]["fuel_matched_"+name] = dict(rows=rows, source=source)
    snapshot.update(generatedAt=now, buildStatus="complete")
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(models=len(models), history=len(history), pairs=len(pairs))))


if __name__ == "__main__":
    main()
