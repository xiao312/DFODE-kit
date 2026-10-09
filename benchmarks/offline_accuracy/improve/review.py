"""Export the complete declared campaign, keeping failures visible as unknowns."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.offline_accuracy.review import primary
from .plan import NAMES, configuration


def collect(root):
    tables = {name: [] for name in ("improve_models", "improve_curves", "improve_history", "improve_bins", "improve_species", "improve_audit")}
    files, identity = [], None
    for seed in (20261011, 20261012):
        for name in NAMES:
            directory = root / f"seed-{seed}" / name
            result = json.loads((directory / "result.json").read_text())
            if result["config"] != configuration(name, seed):
                raise ValueError("Undeclared or changed recipe")
            current = result["hashes"]["dataset_manifest"], result["hashes"]["audit_summary"]
            if identity is not None and identity != current:
                raise ValueError("Mixed data or reference audit")
            identity = current
            key = dict(name=name, seed=seed)
            files.append(dict(name=f"seed-{seed}/{name}/result.json", sha256=sha256(directory / "result.json")))
            if result["status"] == "failed":
                tables["improve_models"].append(dict(key, status="failed", componentRate=None, stateRate=None,
                    trainingComponentRate=None, trainingSeconds=None, inferenceMs=None, negativeRate=None,
                    newlyPassing=None, newlyFailing=None, normalizedErrorP99=None))
                continue
            check = json.loads((directory / "verification.json").read_text())
            if (result["status"] != "complete" or check["status"] != "verified"
                    or check["result_sha256"] != sha256(directory / "result.json")
                    or check["model_replay"] != "exact"
                    or not all(check[field] for field in ("independent_counts", "physical_checks", "training_only_preprocessing",
                                                         "frozen_inputs_unchanged", "transitions_checked"))):
                raise ValueError("Require complete independent verification")
            files.append(dict(name=f"seed-{seed}/{name}/verification.json", sha256=sha256(directory / "verification.json")))
            p, train = primary(result["validation"]), primary(result["training"])
            tables["improve_models"].append(dict(key, status="verified", componentRate=p["component_pass_fraction"],
                stateRate=p["state_pass_fraction"], trainingComponentRate=train["component_pass_fraction"],
                trainingStateRate=train["state_pass_fraction"], trainingSeconds=result["total_training_process_seconds"],
                additionalTrainingSeconds=result["fit_process_seconds"],
                inferenceMs=1000*result["inference"]["median_process_seconds"]/p["states"],
                negativeRate=result["validation_physical"]["negative_endpoint_fraction"],
                correctedRate=result["validation_physical"]["inverse_domain_correction_fraction"],
                newlyPassing=result["validation_transitions"]["fail_to_pass"],
                newlyFailing=result["validation_transitions"]["pass_to_fail"],
                components=p["components"], states=p["states"],
                normalizedErrorP99=p["normalized_error_p99"], parameters=result["total_parameter_count"],
                storageBytes=result.get("table_bytes"), sspiRate=result["validation"]["sspi"]["value"]))
            tables["improve_curves"].extend(dict(key, atol=row["atol"], rtol=row["rtol"], relativeLabel=f'{row["rtol"]*100:g}%',
                componentRate=row["component_pass_fraction"], stateRate=row["state_pass_fraction"]) for row in result["validation"]["tolerances"])
            tables["improve_history"].extend(dict(key, step=row["step"], loss=row["batch_loss"],
                componentRate=row["validation"]["component_pass_fraction"], stateRate=row["validation"]["state_pass_fraction"],
                probeComponentRate=row["training_probe"]["component_pass_fraction"]) for row in result["history"])
            tables["improve_bins"].extend(dict(key, **row) for row in result["validation"]["magnitude_bins"])
            tables["improve_species"].extend(dict(key, **row) for row in result["validation"]["per_species"])
            tables["improve_audit"].append(dict(key, **primary(result["audited_subset"])))
    return tables, files, identity


def extend(snapshot, tables, files, identity):
    contract = snapshot["queries"]["offline_contract"]["rows"][0]
    if identity != (contract["datasetHash"], contract["auditHash"]):
        raise ValueError("Report and campaign identities differ")
    now = datetime.now(timezone.utc).isoformat()
    ids = [f"improve-{name}" for name in ("opening", "summary", "methods", "tolerance", "cost", "table", "transitions", "state-errors", "history", "bins", "audit", "species", "next")]
    for query, rows in tables.items():
        snapshot["queries"][query] = dict(rows=rows, source=dict(
            provider="DFODE-kit independently checked acceptance campaign", name=query, files=files, executedAt=now,
            metricDefinitions=[dict(label="Acceptance", componentIds=ids,
                definition="abs(prediction-reference)<=atol+rtol*abs(reference). Primary atol=1e-15, rtol=0.1; state acceptance requires all 58 non-AR components."),
                dict(label="Base transition", componentIds=["improve-transitions", "improve-table"],
                     definition="New passes and new failures versus the same seed's frozen 4,000-update conventional base, on the same 59,334 scored components."),
                dict(label="Cost", componentIds=["improve-cost", "improve-table"],
                     definition="Total training CPU includes the base for neural adaptations. Local interpolation uses no base prediction. Local solves occur at query time and are included in inference cost. Inference is five-repeat full-batch CPU time divided by state count, not single-query latency.")],
            caveats=["Same-flame development snapshots; no untouched test or CFD claim.",
                     "Rank complete states first, components second. Both seeds and every declared trial retained; no best-seed selection.",
                     "Only 16 development states have independent reference uncertainty checks. Full-population acceptance is nominal.",
                     "Local RBF is not ISAT and has no runtime accuracy certificate. Neural updates and local query solves are not equal-cost protocols."],
            evidenceFlow=[dict(title="Protocol", detail="Frozen data and primary tolerance; seven first-stage methods plus three later Arrhenius-input candidates, both original seeds and fixed final checkpoints. New input candidates change multiple design factors."),
                          dict(title="Verification", detail="Exact saved-model replay; independent physical and acceptance checks; training-only scales, input identities and transition counts.")]))
    snapshot.update(generatedAt=now, buildStatus="updating")
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    tables, files, identity = collect(args.results)
    diagnostic_path = args.results / "state-diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    source_hashes = {row["name"]: row["sha256"] for row in files}
    verified = [row for row in tables["improve_models"] if row["status"] == "verified"]
    expected_sources = {f'seed-{row["seed"]}/{row["name"]}/result.json' for row in verified}
    actual_sources = {row["name"] for row in diagnostic["sources"]}
    if (diagnostic["status"] != "complete" or diagnostic["dataset_manifest_sha256"] != identity[0]
            or expected_sources != actual_sources
            or any(source_hashes.get(row["name"]) != row["sha256"] for row in diagnostic["sources"])):
        raise ValueError("Require matching state-failure diagnostics for every verified model")
    tables["improve_state_errors"] = diagnostic["rows"]
    files.append(dict(name="state-diagnostics.json", sha256=sha256(diagnostic_path)))
    snapshot = extend(json.loads(args.snapshot.read_text(encoding="utf-8")), tables, files, identity)
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(models=len(tables["improve_models"]), sha256=sha256(args.output))))


if __name__ == "__main__":
    main()
