"""Compile the complete verified matrix into the existing report snapshot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from benchmarks.flame_conditioning.extract import sha256
from .plan import TARGETS, OBJECTIVES, POLICIES, configuration


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def primary(policy):
    return next(r for r in policy["grid"] if r["atol"] == 1e-15 and r["rtol"] == .1)


def collect(root):
    rows = {name: [] for name in ("paired_models", "paired_curves", "paired_audit", "paired_bins", "paired_species")}
    files, identities, initial_hashes, warmup_hashes = [], set(), {}, {}
    runtimes = set()
    zero = None
    for seed in (20261011, 20261012):
        for target in TARGETS:
            for objective in OBJECTIVES:
                name = f"{target}-{objective}"
                directory = root / f"seed-{seed}" / name
                result = read(directory / "result.json")
                check = read(directory / "verification.json")
                if (result["status"] != "complete" or result["config"] != configuration(name, seed)
                        or result["updates_completed"] != 4000 or check["status"] != "verified"
                        or check["result_sha256"] != sha256(directory / "result.json")
                        or not all(check.get(k) for k in ("exact_model_replay", "independent_paired_counts",
                                                         "physical_checks", "training_only_preprocessing"))):
                    raise ValueError("Require all twelve complete and verified final GPU fits")
                identities.add((result["hashes"]["dataset_manifest"], result["hashes"]["audit_summary"]))
                runtimes.add(json.dumps(result["runtime"], sort_keys=True))
                initial_hashes.setdefault(seed, set()).add(result["initial_weights_sha256"])
                warmup_hashes.setdefault((seed, target), set()).add(result["warmup_weights_sha256"])
                if zero is not None and zero != result["zero_baseline"]:
                    raise ValueError("Zero control or development population changed")
                zero = result["zero_baseline"]
                for policy in POLICIES:
                    train, dev = primary(result["training"][policy]), primary(result["validation"][policy])
                    if (train["states"], train["components"], dev["states"], dev["components"]) != (10000,580000,1023,59334):
                        raise ValueError("Population differs from the declared comparison")
                    key = dict(name=name, target=target, objective=objective, seed=seed, policy=policy)
                    physical = result["validation_physical"]
                    app = next((r for r in result["validation"][policy]["grid"] if r["role"] == "application-state-diagnostic"), None)
                    rows["paired_models"].append(dict(key, status="verified", trainingComponentRate=train["component_pass_fraction"],
                        trainingStateRate=train["state_pass_fraction"], componentRate=dev["component_pass_fraction"],
                        stateRate=dev["state_pass_fraction"], states=dev["states"], components=dev["components"],
                        componentPassCount=dev["component_pass_count"], statePassCount=dev["state_pass_count"],
                        testRate=None, normalizedP99=dev["normalized_quantiles"]["p99"],
                        applicationComponentRate=None if app is None else app["component_pass_fraction"],
                        applicationStateRate=None if app is None else app["state_pass_fraction"],
                        trainingSeconds=result["training_wall_seconds"], hostCpuSeconds=result["training_process_seconds"],
                        inferenceMs=1000*result["inference"]["median_wall_seconds"]/dev["states"],
                        device=result["runtime"]["device"], peakGpuMiB=result["peak_gpu_bytes"]/1024**2,
                        negativeRate=physical["negative_endpoint_fraction"], correctionRate=physical["inverse_domain_correction_fraction"],
                        massDriftP99=physical["mass_increment_drift"]["p99"]))
                    for item in result["validation"][policy]["grid"]:
                        rows["paired_curves"].append(dict(key, atol=item["atol"], rtol=item["rtol"], role=item["role"],
                            relativeLabel=f'{100*item["rtol"]:g}%', componentRate=item["component_pass_fraction"],
                            stateRate=item["state_pass_fraction"], **item["normalized_quantiles"]))
                    audit = primary(result["audited_subset"][policy])
                    rows["paired_audit"].append(dict(key, **audit["qualification"]))
                    rows["paired_bins"].extend(dict(key, **item) for item in result["validation"][policy]["magnitude_bins"])
                    rows["paired_species"].extend(dict(key, species=item["species"],
                        componentRate=item["component_pass_fraction"], **item["normalized_quantiles"])
                        for item in result["validation"][policy]["per_species"])
                for filename in ("result.json", "verification.json"):
                    files.append(dict(name=f"seed-{seed}/{name}/{filename}", sha256=sha256(directory / filename)))
    if len(identities) != 1 or len(runtimes) != 1 or any(len(v) != 1 for v in [*initial_hashes.values(), *warmup_hashes.values()]):
        raise ValueError("Inputs, initial weights or target-specific warmup differ between matched arms")
    rows["paired_zero"] = [dict(policy=p, **item) for p in POLICIES for item in zero[p]["grid"]]
    return rows, files, identities.pop()


def extend(snapshot, rows, files, identity):
    original = snapshot["queries"]["offline_contract"]["rows"][0]
    if identity != (original["datasetHash"], original["auditHash"]):
        raise ValueError("Existing report and experiment data differ")
    now = datetime.now(timezone.utc).isoformat()
    ids = ["paired-opening", "paired-method", "paired-tolerance", "paired-cost", "paired-table",
           "paired-application", "paired-audit", "paired-bins", "paired-species", "paired-next"]
    for query, values in rows.items():
        snapshot["queries"][query] = dict(rows=values, source=dict(provider="DFODE-kit verified paired experiment",
            name=query, files=files, executedAt=now,
            metricDefinitions=[dict(label="Paired acceptance", componentIds=ids,
                definition="Both policies use abs(predicted increment-reference increment). Divide by atol+rtol*abs(reference increment), or atol+rtol*abs(initial+reference increment). Primary parameters 1e-15 and 0.1. All 58 non-AR species must pass for a state. Reference endpoint, not predicted endpoint. Nominal full-population scores; independent test not evaluated."),
                dict(label="Cost", componentIds=["paired-cost","paired-table"],
                     definition="GPU fit uses synchronized wall seconds including preparation and saving, excluding label generation and independent verification. Inference is synchronized end-to-end median wall time of five full-development-batch calls divided by 1023 states, including host/device transfers and reconstruction; not an online single-query latency. Host CPU seconds are recorded separately, not GPU cost."),
                dict(label="Application-state diagnostic", componentIds=["paired-application"],
                     definition="State-endpoint-v1 with atol=1e-12, rtol=1e-6. Different physical contract, not the training budget or CVODE internal error estimator.")],
            caveats=["Development snapshots from one flame; not an independent-case test.",
                     "GBCT target adaptation, not a paper reproduction. Common train-only mean/RMS normalization.",
                     "Equal updates and architecture; measure actual cost because the losses differ.",
                     "Only sixteen development states have independent reference checks; qualification is an empirical uncertainty screen.",
                     "Inverse-domain corrections are exposed, not hidden. No state renormalization or conservation projection."],
            evidenceFlow=[dict(title="Frozen plan", detail="Two targets, three objectives, two seeds; 2k shared coordinate warmup then 2k objective-specific updates."),
                          dict(title="Verification", detail="Exact model replay, independent paired counts and physical checks; matching dataset, initialization and warmup hashes across arms.")]))
    snapshot["generatedAt"], snapshot["buildStatus"] = now, "updating"
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    rows, files, identity = collect(args.results)
    args.output.write_text(json.dumps(extend(read(args.snapshot), rows, files, identity), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(models=len(rows["paired_models"]), sha256=sha256(args.output))))


if __name__ == "__main__":
    main()
