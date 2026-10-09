"""Append all verified refinements to the existing app without private assets."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.offline_accuracy.review import primary
from .plan import configuration, variants


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root):
    models, curves, histories, audits, bins, species, files = [], [], [], [], [], [], []
    identity = None
    for seed in (20261011, 20261012):
        for variant in variants():
            name = variant["name"]
            directory = root / f"seed-{seed}" / name
            result = read(directory / "result.json")
            check = read(directory / "verification.json")
            if (result["status"] != "complete" or check["status"] != "verified"
                    or check["result_sha256"] != sha256(directory / "result.json")
                    or not check["independent_counts"] or not check["physical_checks"]
                    or not check.get("training_only_scales_checked")
                    or check["model_replay"] != "exact"
                    or result["config"] != configuration(name, seed)
                    or result["updates_completed"] != variant["updates"]):
                raise ValueError("Refuse partial, changed or unverified refinement evidence")
            current = (result["hashes"]["dataset_manifest"], result["hashes"]["audit_summary"])
            if identity is not None and current != identity:
                raise ValueError("Dataset or audit differs across the comparison")
            identity = current
            key = dict(name=name, seed=seed)
            p, train = primary(result["validation"]), primary(result["training"])
            physical = result["validation_physical"]
            models.append(dict(key, target=variant["target"], updates=variant["updates"],
                trainingCount=result["training_count"], states=p["states"], components=p["components"],
                componentRate=p["component_pass_fraction"], stateRate=p["state_pass_fraction"],
                componentPassCount=p["component_pass_count"], statePassCount=p["state_pass_count"],
                trainingComponentRate=train["component_pass_fraction"], trainingStateRate=train["state_pass_fraction"],
                sspiRate=result["validation"]["sspi"]["value"], trainingSeconds=result["total_training_process_seconds"],
                inferenceMs=1000*result["inference"]["median_process_seconds"]/p["states"],
                parameters=result["total_parameter_count"],
                negativeRate=physical["negative_endpoint_fraction"],
                correctionRate=physical["inverse_domain_correction_fraction"],
                massDriftP99=physical["mass_increment_drift"]["p99"],
                residualRoundTripLost=(check["residual_arithmetic"]["nonzero_reconstructed_as_zero"]
                                       if check.get("residual_arithmetic") else None)))
            for item in result["validation"]["tolerances"]:
                curves.append(dict(key, atol=item["atol"], rtol=item["rtol"], relativeLabel=f'{100*item["rtol"]:g}%',
                    componentRate=item["component_pass_fraction"], stateRate=item["state_pass_fraction"]))
            for item in result["history"]:
                histories.append(dict(key, step=item["step"],
                    componentRate=item["validation"]["component_pass_fraction"],
                    stateRate=item["validation"]["state_pass_fraction"],
                    probeComponentRate=item["training_probe"]["component_pass_fraction"],
                    coordinateLoss=item["batch_coordinate_loss"], physicalLoss=item["batch_physical_loss"]))
            audits.append(dict(key, **primary(result["audited_subset"])))
            bins.extend(dict(key, **item) for item in result["validation"]["magnitude_bins"])
            species.extend(dict(key, **item) for item in result["validation"]["per_species"])
            for filename in ("result.json", "verification.json"):
                files.append(dict(name=f"seed-{seed}/{name}/{filename}", sha256=sha256(directory / filename)))
    return dict(refinement_models=models, refinement_curves=curves, refinement_history=histories,
                refinement_audit=audits, refinement_bins=bins, refinement_species=species), files, identity


def extend(snapshot, rows, files, identity):
    contract = snapshot["queries"]["offline_contract"]["rows"][0]
    if identity != (contract["datasetHash"], contract["auditHash"]):
        raise ValueError("Original and new report evidence must use the same data and audit")
    ids = ["refinement-opening", "refinement-summary", "refinement-method", "refinement-tolerance", "refinement-cost",
           "refinement-table", "refinement-history", "refinement-bins", "refinement-audit", "refinement-species", "refinement-next"]
    now = datetime.now(timezone.utc).isoformat()
    for query, values in rows.items():
        snapshot["queries"][query] = dict(rows=values, source=dict(
            provider="DFODE-kit checked offline refinement", name=query, files=files, executedAt=now,
            metricDefinitions=[dict(label="Increment acceptance", componentIds=ids,
                definition="abs(prediction-reference) <= atol + rtol*abs(reference); exclude AR. A state passes all non-AR components. Primary atol=1e-15, rtol=0.1. Scores are nominal outside the audited subset."),
                dict(label="Cost", componentIds=["refinement-cost", "refinement-table"],
                     definition="Process CPU seconds include fit preparation and periodic diagnostics. Residual training adds the original base fit cost. Inference is median of five full-batch calls after warmup, divided by state count, including both networks for residuals. Excludes chemistry labeling and independent verification.")],
            caveats=["Development snapshots from one flame; not independent-case validation or CFD.",
                     "Two fixed seeds, no best-seed selection; longer runs also lengthen the cosine schedule.",
                     "New coordinates use a fixed output scale; original targets use species standardization. This is target-plus-loss conditioning, not an isolated scalar-map effect.",
                     "Deep and residual controls have approximate parameter/work budgets; measured cost is authoritative.",
                     "Reference uncertainty is checked on 16 evaluation states only."],
            evidenceFlow=[dict(title="Replay", detail="Reload final weights; require exact saved prediction replay and independent acceptance/physical checks."),
                          dict(title="Complete matrix", detail="Retain all ten predeclared variants for both seeds, with hash-bound result and verification JSON.")]))
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
    snapshot = extend(read(args.snapshot), rows, files, identity)
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(models=len(rows["refinement_models"]), sha256=sha256(args.output))))


if __name__ == "__main__":
    main()
