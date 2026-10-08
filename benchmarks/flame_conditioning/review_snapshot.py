"""Compile small reviewed evidence for publication without private machine paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.extract import sha256


def score_row(name, values):
    heat_denominator = values["heat_release_reference_rms_W_m3"]
    return {"name": name, "samples": values["samples"],
            "budgetP99": values["budget_error"]["p99"],
            "negativeEndpointFraction": values["negative_endpoint_fraction"],
            "negativeRowFraction": values["negative_endpoint_row_fraction"],
            "inverseCorrectionFraction": values["inverse_domain_correction_fraction"],
            "massDriftP99": values["mass_increment_drift"]["p99"],
            "heatRelativeRms": values["heat_release_error_rms_W_m3"] / heat_denominator if heat_denominator else None}


def reference_row(audit):
    """Keep zero references separate from unresolved nonzero increments."""
    pairs = [(delta, fit) for record in audit["records"]
             for delta, fit in zip(record["reference_delta"], record["relative_fit"], strict=True)]
    if not pairs or len(pairs) != audit["components_checked"]:
        raise ValueError("Reference component counts are inconsistent")
    nonzero = sum(delta != 0 for delta, _ in pairs)
    resolved_nonzero = sum(delta != 0 and fit for delta, fit in pairs)
    return {"states": len(audit["records"]), "speciesComponents": len(pairs),
            "budgetFitFraction": audit["budget_fit_fraction"],
            "relativeFitFraction": sum(fit for _, fit in pairs) / len(pairs),
            "zeroReferenceComponents": len(pairs) - nonzero,
            "nonzeroReferenceComponents": nonzero,
            "resolvedNonzeroComponents": resolved_nonzero,
            "unresolvedNonzeroComponents": nonzero - resolved_nonzero,
            "uncertaintyBudgetMax": max(record["uncertainty_budget_max"] for record in audit["records"])}


def source(files, definitions, caveats):
    return {"provider": "DFODE-kit reproducible experiments", "name": "Saved experiment evidence",
            "files": [{"name": path.name, "sha256": sha256(path)} for path in files],
            "executedAt": datetime.now(timezone.utc).isoformat(),
            "metricDefinitions": definitions, "caveats": caveats,
            "evidenceFlow": [
                {"title": "Read saved evidence", "detail": "Read the listed JSON artifacts and record their SHA256 checksums. No experiment is rerun by this report."},
                {"title": "Select physical metrics", "detail": "Retain model identity, actual training count, precision, update count, validation p99, negative endpoint fraction and heat-release error."},
                {"title": "Preserve limits", "detail": "Keep excluded labels, subset-only reference checks, and offline versus coupled-CFD status distinct."}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cfd", type=Path, required=True)
    parser.add_argument("--parity", type=Path)
    parser.add_argument("--historical-validation", type=Path)
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--heldout-audit", type=Path)
    parser.add_argument("--scaling", type=Path)
    parser.add_argument("--expanded-audit", type=Path)
    parser.add_argument("--dataset-manifest", type=Path, action="append", default=[])
    parser.add_argument("--filter-audit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.heldout_audit and not args.heldout:
        parser.error("A held-out audit must accompany its completed evaluation")
    rows = []
    for path in args.training:
        saved = json.loads(path.read_text())
        if saved["status"] != "complete":
            raise ValueError("Only complete comparisons can enter the completed-results table")
        config = saved["plan"]["config"]
        architecture = "x".join(str(width) for width in config["hidden_widths"])
        for model in saved["variants"]:
            values = model["validation"]
            rows.append({"run": path.parent.name if path.name == "summary.json" else path.stem,
                         "target": model["target"], "precision": model["precision"],
                         "seed": config["seed"],
                         "architecture": architecture, "activation": config.get("activation", "tanh"),
                         "loss": config.get("loss", "mse"), "trainingCount": model["training_count"],
                         "updates": model["updates_completed"], "selectedStep": model["selected_step"],
                         "validationCount": values["samples"], "budgetP99": values["budget_error"]["p99"],
                         "trainingBudgetP99": model["training"]["budget_error"]["p99"],
                         "negativeEndpointFraction": values["negative_endpoint_fraction"],
                         "inverseCorrectionFraction": values["inverse_domain_correction_fraction"],
                         "budgetExceedance": values["budget_exceedance"],
                         "heatRelativeRms": values["heat_release_error_rms_W_m3"] / values["heat_release_reference_rms_W_m3"],
                         "trainingHeatRelativeRms": model["training"]["heat_release_error_rms_W_m3"] / model["training"]["heat_release_reference_rms_W_m3"],
                         "seconds": model["elapsed_seconds"]})
    audit = json.loads(args.audit.read_text())
    if audit.get("reference_subset_pass") is not True:
        raise ValueError("The reference audit did not pass")
    reference = [reference_row(audit)]
    cfd = json.loads(args.cfd.read_text())
    if cfd["status"] != "complete" or not cfd["original_files_unchanged"]:
        raise ValueError("Copied CFD evidence is incomplete")
    snapshot = {"surface": "report", "title": "Flame tests now use the CFD chemistry problem",
                "generatedAt": datetime.now(timezone.utc).isoformat(), "status": "reviewed", "filters": [], "queries": {}}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in ("id", "title", "report", "legacyPresentationTitle"):
            if key in previous:
                snapshot[key] = previous[key]
        snapshot["buildStatus"] = "updating"
    snapshot["queries"]["models"] = {"rows": rows, "source": source(args.training, [
        {"label": "Species budget p99", "definition": "99th percentile over all non-argon species components of abs(predicted increment-reference increment)/(1e-12+1e-6*abs(initial mass fraction)). One is the budget boundary; lower is better."},
        {"label": "Heat-release relative RMS", "definition": "RMS predicted source error divided by RMS reference source; source is -rho/dt times the sum of species formation enthalpy times the species increment. Lower is better; 1 is the zero-increment baseline."},
        {"label": "Negative endpoint fraction", "definition": "Fraction of non-argon species components for which initial Y plus predicted increment is negative. No post-hoc normalization or positivity repair except the separately counted Box-Cox inverse-domain correction."}
    ], ["Validation snapshots come from the same 1D flame realization.", "The main four-target comparisons use one seed. A conventional-only fixed second-seed repeat, when present, is a sensitivity check, not a seed search or statistical ranking.", "The strict species budget is a research criterion, not a guarantee of CFD solver accuracy."])}
    snapshot["queries"]["reference"] = {"rows": reference, "source": source([args.audit], [
        {"label": "Reference agreement", "definition": "Maximum spread across stored CVODE, fresh tighter and step-limited CVODE, and two independent Radau increment integrations, divided by the species budget. This is empirical agreement, not a rigorous bound."},
        {"label": "Relative-resolution screen", "definition": "Absolute reference increment must exceed 100 times the larger of empirical solver disagreement and endpoint spacing. Zero reference increments and unresolved nonzero increments are reported separately; zero references are not claims of exact mathematical zero."}
    ], ["Only 32 selected augmented states are independently audited; remaining labels are not individually certified.",
        "Agreement applies to the defined rounded/normalized source inputs; FP64 storage and tighter integration do not restore original ASCII digits."])}
    snapshot["queries"]["cfd"] = {"rows": [{**state, "steps": cfd["steps"], "originalFilesUnchanged": cfd["original_files_rechecked"]} for state in cfd["states"]],
        "source": source([args.cfd], [{"label": "Copied CFD restart", "definition": "Read-only review of 500-cell initial/final scalar fields, completed solver log, mesh check, and all 73 original allowlisted file hashes."}],
                         ["CVODE-only run with neural chemistry disabled. No learned model was deployed.", "Installed CFD uses Cantera 2.6.0; research labels use 3.2.0. Existing runtime was not upgraded."])}
    if args.parity:
        parity = json.loads(args.parity.read_text())
        if parity["status"] != "complete":
            raise ValueError("Runtime parity evidence is incomplete")
        snapshot["queries"]["runtime_parity"] = {"rows": [{"states": len(parity["records"]),
            "runtimeCantera": parity["plan"]["runtime_cantera"], "researchCantera": parity["plan"]["research_cantera"],
            "maxDifferenceBudget": parity["max_difference_budget"], "budgetFitFraction": parity["budget_fit_fraction"]}],
            "source": source([args.parity], [{"label": "Version/interface parity", "definition": "Difference between fixed-T/V tight Cantera 2.6 Reactor increments and saved Cantera 3.2 IdealGasReactor labels, divided by the species budget."}],
                             ["Selected validation subset only; not a guarantee across every state or all solver settings."])}
    if cfd.get("profiles"):
        snapshot["queries"]["flame_profile"] = {"rows": cfd["profiles"], "source": source([args.cfd], [
            {"label": "Temperature profile", "definition": "Initial and final cell temperature versus actual cell-centre x position. Nonuniform mesh coordinates are checked against the original geometry-vector field; the mesh is not assumed uniform."}
        ], ["The 0.1 ms CVODE-only restart is a compatibility check, not a flame-speed or steady-state validation."])}
    if args.historical_validation:
        historical = json.loads(args.historical_validation.read_text())
        if historical["status"] != "complete":
            raise ValueError("Historical comparison is incomplete")
        snapshot["queries"]["historical"] = {"rows": [
            {**score_row(model["name"], model["validation"]),
             "inverseDomainViolationFraction": model.get("diagnostics", {}).get("inverse_domain_violation_fraction")}
            for model in historical["models"]],
            "source": source([args.historical_validation], [
                {"label": "Historical-weight control", "definition": "Preselected existing FP32 weights, checked and converted to plain numerical arrays. Both original-style and stable reconstruction use the same weights; the hybrid uses fixed 305/1000 K thresholds."}
            ], ["Historical training overlap with this domain is not excluded; this is not independent generalization evidence.",
                "Historical training used far more data and updates. Do not interpret this as a matched-compute comparison.",
                "Saved direct-power statistics retain FP32 only. Exact original FP64 preprocessing is unavailable."])}
    if args.heldout:
        heldout = json.loads(args.heldout.read_text())
        if heldout["status"] != "complete":
            raise ValueError("Reserved-snapshot scoring is incomplete")
        test_rows = []
        for model in heldout["models"]:
            timing = model.get("prediction_timing") or {}
            for population, values in model["populations"].items():
                test_rows.append({**score_row(model["name"], values), "population": population,
                                  "historical": model["name"].startswith("historical--"),
                                  "batchStates": timing.get("states"),
                                  "batchWallSeconds": timing.get("wall_seconds"),
                                  "batchProcessSeconds": timing.get("process_seconds"),
                                  "batchInverseDomainViolationFraction": model.get("diagnostics", {}).get("inverse_domain_violation_fraction")})
        snapshot["queries"]["heldout"] = {"rows": test_rows, "source": source([args.heldout], [
            {"label": "Reserved 2D snapshot", "definition": "Offline predictions on a predeclared uniform-cell sample and a separate temperature-balanced diagnostic sample. Model hashes and thresholds were fixed before reading the snapshot. No post-test tuning."}
        ], ["One snapshot is not a coupled CFD trajectory or a statistical generalization study.",
            "Uniform and temperature-balanced samples overlap; do not pool them or average their scores.",
            "Timing covers one combined offline batch, not each population separately or a CFD speedup.",
            "Historical raw inverse-domain violations are combined-batch diagnostics when recorded. Zero corrections do not imply zero invalid bases; null means not recorded.",
            "Historical models may have prior training/evaluation exposure; only newly trained controls were kept from this snapshot."])}
        snapshot["queries"]["heldout_sampling"] = {"rows": [
            {"population": name, **counts, "excluded": counts["selected"] - counts["accepted"]}
            for name, counts in heldout["sample_counts"].items()],
            "source": source([args.heldout], [{"label": "Test label acceptance", "definition": "Selected cell counts and accepted fixed-T/V labels, reported separately for the overlapping uniform and balanced populations."}],
                             ["Do not add the two population counts; some cells occur in both."])}
        if args.heldout_audit:
            audit = json.loads(args.heldout_audit.read_text())
            if (heldout["audit_summary_sha256"] != sha256(args.heldout_audit)
                    or audit["status"] != "complete" or audit["budget_fit_fraction"] != 1.0):
                raise ValueError("Held-out audit is incomplete, failed, or differs from the scored audit")
            snapshot["queries"]["heldout_reference"] = {"rows": [reference_row(audit)],
                "source": source([args.heldout_audit], [{"label": "Test reference agreement", "definition": "Selected reserved-snapshot states checked by tighter/step-limited CVODE and direct-increment Radau, before model scoring. Zero and unresolved nonzero references remain separate."}],
                                 ["Subset empirical agreement, not certification of all test labels or many significant digits."])}
    if args.scaling:
        scaling = json.loads(args.scaling.read_text())
        if scaling["status"] != "verified":
            raise ValueError("Data-size comparison is not verified")
        snapshot["queries"]["scaling"] = {"rows": [scaling], "source": source([args.scaling], [
            {"label": "Data-size identity", "definition": "Same raw training prefix, identical validation inputs and accepted masks, matching source/configuration except training count and wall limit, and stored label differences below 0.01 species budget."}
        ], ["Equal update counts give the larger dataset fewer average presentations per training state."])}
    if args.expanded_audit:
        expanded = json.loads(args.expanded_audit.read_text())
        if expanded.get("reference_subset_pass") is not True:
            raise ValueError("Expanded-data reference audit did not pass")
        snapshot["queries"]["expanded_reference"] = {"rows": [{"states": len(expanded["records"]),
            "components": expanded["components_checked"], "budgetFitFraction": expanded["budget_fit_fraction"],
            "uncertaintyBudgetMax": max(row["uncertainty_budget_max"] for row in expanded["records"])}],
            "source": source([args.expanded_audit] + ([args.scaling] if args.scaling else []), [{"label": "Expanded reference agreement", "definition": "Same independent endpoint/increment checks on a selected subset of the larger dataset. The optional scaling evidence verifies prefix and validation identity."}],
                             ["Subset agreement does not certify every generated label."])}
    if args.dataset_manifest:
        dataset_rows = []
        for path in args.dataset_manifest:
            dataset = json.loads(path.read_text())
            if dataset["status"] not in ("complete", "complete_with_exclusions"):
                raise ValueError("Dataset generation is incomplete")
            dataset_rows.append({"candidates": dataset["config"]["train_count"],
                                 "trainingAccepted": dataset["splits"]["train"]["labels_accepted"],
                                 "validationAccepted": dataset["splits"]["validation"]["labels_accepted"],
                                 "generationSeconds": dataset["elapsed_seconds"],
                                 "continuations": len(dataset.get("continuations", []))})
        snapshot["queries"]["datasets"] = {"rows": dataset_rows, "source": source(args.dataset_manifest, [
            {"label": "Accepted chemistry labels", "definition": "Candidates with finite, nonnegative endpoints and passing fixed-temperature/volume, mass and elemental checks. Failed rows remain in the private artifact but are excluded from fitting."}
        ], ["The larger generation run reached its first wall limit and finished in a new continuation directory; its elapsed time is the sum of both segments."])}
    if args.filter_audit:
        filter_rows = []
        for path in args.filter_audit:
            audit = json.loads(path.read_text())
            if audit.get("status") != "complete":
                raise ValueError("Historical-filter audit is incomplete")
            for split, counts in audit["splits"].items():
                filter_rows.append({"trainingCount": audit["splits"]["train"]["states"],
                                    "split": split, "states": counts["states"],
                                    "kept": counts["kept"], "rejected": counts["rejected"],
                                    "rejectedFraction": counts["rejected_fraction"],
                                    "rule": audit["rule"]})
        snapshot["queries"]["filter_audit"] = {"rows": filter_rows, "source": source(args.filter_audit, [
            {"label": "Historical heat-release filter", "definition": "Keep sum(hf_298 * delta_Y) <= 200 J/kg. Positive sums mean endothermic change. This rejects sufficiently endothermic states, not all negative heat release."}
        ], ["Read-only diagnostic: current labels and fitting data are unchanged.",
            "Removing validation rows is not equivalent to retraining on filtered data."])}
    if not args.dry_run:
        args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False))
    print(json.dumps({"dry_run": args.dry_run, "output": str(args.output), "model_rows": len(rows), "queries": list(snapshot["queries"])}))


if __name__ == "__main__":
    main()
