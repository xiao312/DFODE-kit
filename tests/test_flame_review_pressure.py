import copy

import pytest

from benchmarks.flame_conditioning.review_pressure import pressure_rows


def test_pressure_review_binds_verified_sources_and_keeps_both_conditions():
    scores = {"budget_error": {"p99": 12.}, "heat_release_error_rms_W_m3": 2.,
              "heat_release_reference_rms_W_m3": 4., "negative_endpoint_fraction": .1}
    diagnostic = {"status": "complete", "plan": {"count": 32, "pressure_Pa": 101414.,
        "test_manifest_sha256": "test", "training_summary_sha256": "train", "audit_summary_sha256": "audit"},
        "reference_heat_response_relative_rms": .01, "uncertainty_budget_max": 1e-5,
        "models": [{"target": "state-boxcox", "prediction_heat_response_relative_rms": .3,
                    "conditions": {"original": scores, "training_mean_pressure": scores}}]}
    verified = {"status": "verified", "diagnostic_summary_sha256": "diagnostic", "test_manifest_sha256": "test",
                "original_inputs_and_labels_unchanged": True, "only_pressure_changed": True,
                "reference_records_verified": 32}
    heldout = {"test_manifest_sha256": "test", "audit_summary_sha256": "audit"}
    rows = pressure_rows(diagnostic, verified, heldout, {"train"}, "diagnostic")
    assert rows[0]["originalHeatRelativeRms"] == rows[0]["changedHeatRelativeRms"] == .5
    assert rows[0]["states"] == 32 and rows[0]["originalBudgetP99"] == 12.
    for key in ("original_inputs_and_labels_unchanged", "only_pressure_changed"):
        bad = copy.deepcopy(verified)
        bad[key] = False
        with pytest.raises(ValueError):
            pressure_rows(diagnostic, bad, heldout, {"train"}, "diagnostic")
    with pytest.raises(ValueError):
        pressure_rows(diagnostic, verified, heldout, {"other-training"}, "diagnostic")
    with pytest.raises(ValueError):
        pressure_rows(diagnostic, verified, heldout, {"train"}, "wrong-diagnostic")
