"""Append verified fixed-data budget evidence to the same review artifact."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256
from ..review import primary
from .review import collect as collect_baseline, validate_result, POLICIES


def pairs_from_rows(rows):
    pairs = []
    for recipe in ("fuel-state-budget", "fuel-power-budget"):
        for seed in (20261011, 20261012):
            for policy in POLICIES:
                pair = sorted((r for r in rows if (r["recipe"],r["seed"],r["policy"]) == (recipe,seed,policy)), key=lambda r:r["updates"])
                if [r["updates"] for r in pair] != [6000,18000]:
                    raise ValueError("Need exactly one result at both budgets")
                small, large = pair
                pairs.append(dict(recipe=recipe, seed=seed, policy=policy,
                    smallRate=small["developmentRate"], largeRate=large["developmentRate"],
                    changePoints=100*(large["developmentRate"]-small["developmentRate"]),
                    wallRatio=large["fitSeconds"]/small["fitSeconds"]))
    return pairs


def collect(root, baseline):
    old_models, _, _, files, _ = collect_baseline(baseline)
    files = [dict(name="baseline/"+row["name"],sha256=row["sha256"]) for row in files]
    check = json.loads((root / "verification.json").read_text())
    if (check["status"] != "verified" or not all(check.get(k) for k in
            ("fixed_data","common_normalization","identical_initial_weights"))
            or check["baseline_verification_sha256"] != sha256(baseline / "verification.json")):
        raise ValueError("Missing or changed campaign verification")
    bound = {r["name"]:r["result_sha256"] for r in check["results"]}
    if len(bound) != 4 or len(check["results"]) != 4:
        raise ValueError("Need all four new fits")
    models, history = [], []
    for seed in (20261011,20261012):
        for recipe in ("fuel-state","fuel-power"):
            old_path = baseline / f"{seed}-{recipe}-n200000"
            old = json.loads((old_path / "result.json").read_text())
            new_path = root / f"{seed}-{recipe}-u18000"
            new = json.loads((new_path / "result.json").read_text())
            if (new["baseline_result_sha256"] != sha256(old_path / "result.json")
                    or bound.get(new_path.name) != sha256(new_path / "result.json")):
                raise ValueError("Baseline or new result hash differs")
            for key in ("hashes","initial_weights_sha256","preprocessing_array_sha256"):
                if old[key] != new[key]:
                    raise ValueError("Fixed-data identity differs")
            for path, result, updates in ((old_path,old,6000),(new_path,new,18000)):
                verification = json.loads((path / "verification.json").read_text())
                validate_result(result, verification, sha256(path / "result.json"), sha256(path / "environment.json"),
                                recipe, seed, 200000, extended=updates == 18000)
                for policy in POLICIES:
                    train, dev = primary(result["training"][policy]), primary(result["development"][policy])
                    models.append(dict(recipe=recipe+"-budget", seed=seed, policy=policy, updates=updates,
                        trainingCount=200000, presentations=result["row_presentations"],
                        trainingRate=train["component_pass_fraction"], developmentRate=dev["component_pass_fraction"],
                        stateRate=dev["state_pass_fraction"], p99=dev["normalized_quantiles"]["p99"],
                        zeroRate=primary(result["zero_baseline"][policy])["component_pass_fraction"],
                        zeroP99=primary(result["zero_baseline"][policy])["normalized_quantiles"]["p99"],
                        fitSeconds=result["training_wall_seconds"],
                        inferenceMicrosecondsPerState=1e6*result["inference"]["median_wall_seconds"]/result["inference"]["states"],
                        negativeRate=result["development_physical"]["negative_endpoint_fraction"],
                        correctionRate=result["development_physical"]["inverse_domain_correction_fraction"], independentTest=None))
                    for record in result["history"]:
                        if record["row_presentations"] != record["updates"]*10000:
                            raise ValueError("History work differs")
                        for split in ("training","development"):
                            history.append(dict(recipe=recipe+"-budget",seed=seed,policy=policy,budget=updates,
                                split=split,series=f"{updates//1000}k budget: {split}",updates=record["updates"],
                                acceptanceRate=record[split][policy]["component_pass_fraction"],seconds=record["elapsed_wall_seconds"]))
            for name in ("result.json","verification.json","environment.json"):
                files.append(dict(name=f"extended/{new_path.name}/{name}",sha256=sha256(new_path / name)))
    files.append(dict(name="extended/verification.json",sha256=sha256(root / "verification.json")))
    return models, history, pairs_from_rows(models), files, old_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    models, history, pairs, files, old_models = collect(args.campaign,args.baseline)
    if snapshot["queries"]["fuel_matched_models"]["rows"] != old_models:
        raise ValueError("Published baseline and experiment baseline differ")
    now = datetime.now(timezone.utc).isoformat()
    source = dict(type="file",executedAt=now,files=files,
        evidenceFlow=[dict(title="Checked budget comparison",detail="Require the completed four-fit campaign, its verified eight-fit baseline, bound result/environment hashes, exact work schedules and paired initialization/preprocessing/source identities. Remote verification checks saved arrays, model replay and independent physical metrics. Extract small summaries only; do not read the independent test.")],
        metricDefinitions=[dict(label="Fixed-data accuracy and work",componentIds=["budget-opening","budget-protocol","budget-pairs","budget-models","budget-history","budget-next"],
            definition="Both budgets use the same 200k training rows, 1023 development states, 58 species and frozen 50k scalers. Primary allowance: 1e-15+0.1*abs(reference increment), or reference endpoint for state policy. Same signed-increment error. Whole-state pass requires all components. Zero control predicts zero increment. Change is in percentage points. Fit wall includes full-pool diagnostics; inference is median of five checked 1023-state GPU batches, including preprocessing, transfers and FP64 reconstruction, divided by states; not isolated single-state latency.")],
        caveats=["Fresh fits with proportionally stretched learning-rate schedules, not checkpoint continuation.","Both seeds, final checkpoint only. Correlated same-flame development, not independent-case testing.","Nominal reference metrics; subset convergence checks do not certify every label. No solver-level claim."])
    for name, rows in (("models",models),("history",history),("pairs",pairs)):
        snapshot["queries"]["fuel_budget_"+name] = dict(rows=rows,source=source)
    snapshot.update(generatedAt=now,buildStatus="complete")
    args.output.write_text(json.dumps(snapshot,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(dict(models=len(models),history=len(history),pairs=len(pairs))))


if __name__ == "__main__":
    main()
