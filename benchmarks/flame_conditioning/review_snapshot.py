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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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
                         "architecture": architecture, "activation": config.get("activation", "tanh"),
                         "loss": config.get("loss", "mse"), "trainingCount": model["training_count"],
                         "updates": model["updates_completed"], "selectedStep": model["selected_step"],
                         "validationCount": values["samples"], "budgetP99": values["budget_error"]["p99"],
                         "trainingBudgetP99": model["training"]["budget_error"]["p99"],
                         "negativeEndpointFraction": values["negative_endpoint_fraction"],
                         "inverseCorrectionFraction": values["inverse_domain_correction_fraction"],
                         "budgetExceedance": values["budget_exceedance"],
                         "heatRelativeRms": values["heat_release_error_rms_W_m3"] / values["heat_release_reference_rms_W_m3"],
                         "seconds": model["elapsed_seconds"]})
    audit = json.loads(args.audit.read_text())
    if audit.get("reference_subset_pass") is not True:
        raise ValueError("The reference audit did not pass")
    reference = [{"states": len(audit["records"]), "speciesComponents": audit["components_checked"],
                  "budgetFitFraction": audit["budget_fit_fraction"], "relativeFitFraction": audit["relative_fit_fraction"],
                  "uncertaintyBudgetMax": max(record["uncertainty_budget_max"] for record in audit["records"])}]
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
    ], ["Validation snapshots come from the same 1D flame realization.", "One seed; this is not a statistical ranking or exact paper reproduction.", "The strict species budget is a research criterion, not a guarantee of CFD solver accuracy."])}
    snapshot["queries"]["reference"] = {"rows": reference, "source": source([args.audit], [
        {"label": "Reference agreement", "definition": "Maximum spread across stored CVODE, fresh tighter and step-limited CVODE, and two independent Radau increment integrations, divided by the species budget. This is empirical agreement, not a rigorous bound."}
    ], ["Only 32 selected augmented states are independently audited; remaining labels are not individually certified."])}
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
    if not args.dry_run:
        args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False))
    print(json.dumps({"dry_run": args.dry_run, "output": str(args.output), "model_rows": len(rows), "queries": list(snapshot["queries"])}))


if __name__ == "__main__":
    main()
