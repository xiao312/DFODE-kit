"""Add only verified Fuel source-recipe summaries to the existing review snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from benchmarks.flame_conditioning.extract import sha256
from .plan import configuration


def primary(summary):
    return next(row for row in summary["grid"] if row["atol"] == 1e-15 and row["rtol"] == .1)


def collect(root, training_count=10000):
    models, history, files = [], [], []
    identity = set()
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            directory = root / f"seed-{seed}" / recipe
            result = json.loads((directory / "result.json").read_text())
            check = json.loads((directory / "verification.json").read_text())
            runtime = json.loads((directory / "environment.json").read_text())
            batch = min(training_count, 20000)
            batches = training_count//batch if recipe == "fuel-state" else (training_count+batch-1)//batch
            presentations = batches*batch if recipe == "fuel-state" else training_count
            if (result["status"] != "complete" or check["status"] != "verified"
                    or check["result_sha256"] != sha256(directory / "result.json")
                    or result["artifacts"]["environment.json"] != sha256(directory / "environment.json")
                    or result["config"] != configuration(recipe, seed)
                    or result["training_count"] != training_count or result["development_count"] != 1023
                    or result["epochs_completed"] != result["config"]["epochs"]
                    or result["updates_completed"] != result["epochs_completed"]*batches
                    or result["row_presentations"] != result["epochs_completed"]*presentations
                    or result["parameter_count"] != 2018458
                    or not all(check.get(k) for k in ("exact_model_replay", "independent_paired_counts",
                                                     "physical_checks", "training_only_preprocessing"))):
                raise ValueError("Need all four verified fits with the declared training count and work budget")
            identity.add((result["hashes"]["dataset_manifest"], result["hashes"]["audit_summary"]))
            for policy in ("increment-reference-v1", "state-endpoint-v1"):
                train, dev = primary(result["training"][policy]), primary(result["development"][policy])
                models.append(dict(recipe=recipe, seed=seed, policy=policy, trainingCount=training_count,
                    trainingRate=train["component_pass_fraction"], developmentRate=dev["component_pass_fraction"],
                    stateRate=dev["state_pass_fraction"], p99=dev["normalized_quantiles"]["p99"],
                    epochs=result["epochs_completed"], updates=result["updates_completed"],
                    presentations=result["row_presentations"], fitSeconds=result["training_wall_seconds"],
                    sspi=result["development_sspi"]["value"],
                    correctionRate=result["development_physical"]["inverse_domain_correction_fraction"],
                    negativeRate=result["development_physical"]["negative_endpoint_fraction"],
                    device=runtime["device"], independentTest=None))
                for record in result["history"]:
                    for split in ("training", "development"):
                        history.append(dict(recipe=recipe, seed=seed, policy=policy, split=split,
                            epoch=record["epoch"], presentations=record["row_presentations"],
                            acceptanceRate=record[split][policy]["component_pass_fraction"],
                            p99=record[split][policy]["normalized_quantiles"]["p99"],
                            seconds=record["elapsed_wall_seconds"]))
            for name in ("result.json", "verification.json", "environment.json"):
                files.append(dict(name=f"seed-{seed}/{recipe}/{name}", sha256=sha256(directory / name)))
    if len(identity) != 1:
        raise ValueError("Comparison identities differ")
    return models, history, files, identity.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    models, history, files, identity = collect(args.campaign)
    contract = snapshot["queries"]["offline_contract"]["rows"][0]
    if identity != (contract["datasetHash"], contract["auditHash"]):
        raise ValueError("Existing report and Fuel run data differ")
    now = datetime.now(timezone.utc).isoformat()
    for name, rows in (("fuel_models", models), ("fuel_history", history)):
        snapshot["queries"][name] = dict(rows=rows, source=dict(provider="DFODE-kit verified GPU experiments",
            name=name, executedAt=now, files=files,
            metricDefinitions=[dict(label="Acceptance", componentIds=["fuel-opening", "fuel-results", "fuel-history", "fuel-next"],
                definition="Same signed-increment error; primary a=1e-15,r=0.1. Increment policy scales by abs(reference increment); state policy by abs(reference endpoint). 10000 training and 1023 development states, 58 scored species. All components must pass for a complete state. Nominal scores; independent test not evaluated."),
                dict(label="Fit cost and exposure", componentIds=["fuel-results", "fuel-history", "fuel-next"],
                     definition="Synchronized GPU wall time includes preprocessing and fixed-epoch diagnostics, excludes reference generation and independent verification. Row presentations count actual batches; not unique samples. Full-batch 10k source-recipe adaptation."),
                dict(label="SSPI", componentIds=["fuel-results"],
                     definition="Fraction of reference increments with abs(d)<1e-15 whose predictions also have magnitude<1e-15. Not the mixed-budget pass fraction.")],
            caveats=["Reduced-data saved-source recipe, not full paper reproduction.",
                     "Fixed final checkpoint; both seeds retained. One flame development domain, not independent-case testing.",
                     "Source recipes change several factors together; this is not an isolated normalization ablation.",
                     "Stable FP64 inverse, train-only statistics and explicit invalid-domain corrections are deliberate source-code deviations."] ))
    snapshot["generatedAt"] = now
    snapshot["buildStatus"] = "complete"
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(models=len(models), history=len(history), original_queries_preserved=True)))


if __name__ == "__main__":
    main()
