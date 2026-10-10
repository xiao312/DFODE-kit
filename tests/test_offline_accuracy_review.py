import copy
import json
import numpy as np
import pytest
from benchmarks.offline_accuracy.review import extend_snapshot, compile_rows
from benchmarks.offline_accuracy.metrics import summarize
from benchmarks.flame_conditioning.extract import sha256


def test_review_preserves_identity_prior_queries_and_nulls():
    original = {"id": "stable", "title": "Existing title", "report": {"asOf": "2026-10-08"},
                "queries": {"prior": {"rows": [{"unchanged": 1}]}}}
    result = extend_snapshot(copy.deepcopy(original), {"offline_models": [{"rate": None}]}, [])
    assert result["id"] == original["id"] and result["title"] == original["title"]
    assert result["queries"]["prior"] == original["queries"]["prior"]
    assert result["queries"]["offline_models"]["rows"][0]["rate"] is None
    assert result["buildStatus"] == "updating"


def test_review_rejects_missing_seeds():
    with pytest.raises(ValueError, match="both predeclared seeds"):
        compile_rows([])


def test_existing_report_without_as_of_metadata_remains_supported():
    snapshot = {"id": "old-report", "queries": {}}
    result = extend_snapshot(snapshot, {}, [])
    assert result["report"]["asOf"] == "2026-10-09"
    assert result["id"] == "old-report"


def fixture_seed(root, seed):
    """Synthetic verifier fixtures only; never report these as experiment data."""
    directory = root / str(seed)
    (directory / "training").mkdir(parents=True)
    (directory / "evaluation").mkdir()
    targets = ["state-boxcox", "signed-power", "budget-linear", "scaled-asinh"]
    config = {"seed": seed, "updates": 2000, "checkpoint_selection": "final",
              "training_sizes": [2000, 10000], "hidden_widths": [800]*4,
              "batch_size": 256, "precisions": ["float32"], "loss": "l1",
              "activation": "gelu", "targets": targets}
    training_path = directory / "training/summary.json"
    training_path.write_text(json.dumps({"status": "complete", "plan": {"config": config}}))
    values = summarize(np.ones((2, 2)), np.ones((2, 2)), ["A", "B"])
    models = [{"target": target, "training_count": size, "nominal": values, "audited_subset": values,
               "physical": {"negative_endpoint_fraction": 0, "mass_increment_drift": {"p99": 0},
                            "inverse_domain_correction_fraction": 0},
               "training_process_seconds": 3., "training_wall_seconds": 6.,
               "inference": {"median_process_seconds": .01}}
              for target in targets for size in (2000, 10000)]
    evaluation_path = directory / "evaluation/summary.json"
    evaluation_path.write_text(json.dumps({"status": "complete", "plan": {"seed": seed, "models": 8},
        "training_summary_sha256": sha256(training_path), "dataset_manifest_sha256": "same-data",
        "audit_summary_sha256": "same-audit", "zero_baseline": {"nominal": values},
        "models": models, "dataset": {}}))
    (directory / "replay.json").write_text(json.dumps({"status": "verified", "training_metrics_checked": True,
                                                       "training_summary_sha256": sha256(training_path)}))
    (directory / "acceptance-verification.json").write_text(json.dumps({"status": "verified",
        "evaluation_summary_sha256": sha256(evaluation_path)}))
    return directory


def test_review_keeps_all_fits_and_zero_curves(tmp_path):
    paths = [fixture_seed(tmp_path, seed) for seed in (20261011, 20261012)]
    rows, files = compile_rows(paths)
    assert len(rows["offline_models"]) == 16
    assert len(rows["offline_curves"]) == 16*12 + 12
    assert len(files) == 8
    assert {row["seed"] for row in rows["offline_models"]} == {20261011, 20261012}
    assert sum(row["target"] == "zero" for row in rows["offline_curves"]) == 12


def test_review_rejects_modified_checked_summary(tmp_path):
    paths = [fixture_seed(tmp_path, seed) for seed in (20261011, 20261012)]
    path = paths[0] / "evaluation/summary.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="hash-bound"):
        compile_rows(paths)
