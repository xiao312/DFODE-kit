"""Bind a post-score pressure diagnostic to its checked sources for review."""


def pressure_rows(diagnostic, verified, heldout, training_hashes, diagnostic_hash):
    plan = diagnostic["plan"]
    if (diagnostic["status"] != "complete" or verified["status"] != "verified"
            or verified["diagnostic_summary_sha256"] != diagnostic_hash
            or plan["test_manifest_sha256"] != heldout["test_manifest_sha256"]
            or verified["test_manifest_sha256"] != plan["test_manifest_sha256"]
            or plan["training_summary_sha256"] not in training_hashes
            or plan["audit_summary_sha256"] != heldout["audit_summary_sha256"]
            or not verified["original_inputs_and_labels_unchanged"] or not verified["only_pressure_changed"]
            or verified["reference_records_verified"] != plan["count"]):
        raise ValueError("Pressure diagnostic must match verified training, audit, and test evidence")
    rows = []
    for model in diagnostic["models"]:
        row = {"target": model["target"], "states": plan["count"], "pressurePa": plan["pressure_Pa"],
               "referenceHeatResponse": diagnostic["reference_heat_response_relative_rms"],
               "modelHeatResponse": model["prediction_heat_response_relative_rms"],
               "referenceUncertaintyBudgetMax": diagnostic["uncertainty_budget_max"]}
        for condition, prefix in (("original", "original"), ("training_mean_pressure", "changed")):
            scores = model["conditions"][condition]
            row[f"{prefix}BudgetP99"] = scores["budget_error"]["p99"]
            row[f"{prefix}HeatRelativeRms"] = scores["heat_release_error_rms_W_m3"] / scores["heat_release_reference_rms_W_m3"]
            row[f"{prefix}NegativeFraction"] = scores["negative_endpoint_fraction"]
        rows.append(row)
    return rows
