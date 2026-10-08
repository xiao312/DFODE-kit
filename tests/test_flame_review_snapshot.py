import json
import sys

import pytest

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.review_snapshot import main, reference_row


def test_reference_report_separates_zero_and_unresolved_nonzero():
    audit = {"components_checked": 4, "budget_fit_fraction": 1.0, "records": [
        {"reference_delta": [0.0, 1e-12, -1e-20, 1e-32],
         "relative_fit": [False, True, True, False], "uncertainty_budget_max": 1e-5}]}
    row = reference_row(audit)
    assert row["zeroReferenceComponents"] == 1
    assert row["nonzeroReferenceComponents"] == 3
    assert row["resolvedNonzeroComponents"] == 2
    assert row["unresolvedNonzeroComponents"] == 1
    assert row["relativeFitFraction"] == .5
    audit["components_checked"] = 5
    with pytest.raises(ValueError):
        reference_row(audit)


def test_review_binds_test_reference_and_keeps_population_counts(tmp_path, monkeypatch):
    values = {"samples": 1, "budget_error": {"p99": 2.0}, "negative_endpoint_fraction": 0.0,
              "negative_endpoint_row_fraction": 0.0, "inverse_domain_correction_fraction": 0.0,
              "mass_increment_drift": {"p99": 0.0}, "heat_release_reference_rms_W_m3": 2.0,
              "heat_release_error_rms_W_m3": 1.0, "budget_exceedance": {}}
    training = {"status": "complete", "plan": {"config": {"hidden_widths": [2]}}, "variants": [
        {"target": "state-boxcox", "precision": "float32", "training_count": 1,
         "updates_completed": 1, "selected_step": 1, "elapsed_seconds": .1,
         "training": values, "validation": values}]}
    audit = {"status": "complete", "reference_subset_pass": True, "components_checked": 1,
             "budget_fit_fraction": 1.0, "records": [
                 {"reference_delta": [0.0], "relative_fit": [False], "uncertainty_budget_max": 0.0}]}
    cfd = {"status": "complete", "original_files_unchanged": True, "steps": 1,
           "original_files_rechecked": 1, "states": []}
    for name, data in (("training", training), ("audit", audit), ("cfd", cfd)):
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    heldout = {"status": "complete", "audit_summary_sha256": sha256(tmp_path / "audit.json"),
               "sample_counts": {"uniform": {"selected": 2, "accepted": 1}},
               "models": [{"name": "zero-baseline", "populations": {"uniform": values}}]}
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(json.dumps(heldout))
    output = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", ["review_snapshot", "--training", str(tmp_path / "training.json"),
        "--audit", str(tmp_path / "audit.json"), "--cfd", str(tmp_path / "cfd.json"),
        "--heldout", str(heldout_path), "--heldout-audit", str(tmp_path / "audit.json"),
        "--output", str(output)])
    main()
    text = output.read_text()
    queries = json.loads(text)["queries"]
    assert queries["heldout_sampling"]["rows"][0]["excluded"] == 1
    assert queries["heldout_reference"]["rows"][0]["zeroReferenceComponents"] == 1
    assert queries["heldout"]["rows"][0]["batchWallSeconds"] is None
    assert str(tmp_path) not in text
    heldout["audit_summary_sha256"] = "wrong"
    heldout_path.write_text(json.dumps(heldout))
    with pytest.raises(ValueError, match="differs from the scored audit"):
        main()
