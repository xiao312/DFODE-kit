"""Synthetic export fixtures only; never use these rows as research evidence."""
import copy
import json

import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.offline_accuracy.improve.plan import NAMES, configuration
from benchmarks.offline_accuracy.improve.review import collect, extend, physics_tables


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    primary = dict(atol=1e-15, rtol=.1, states=2, components=4,
                   component_pass_fraction=.25, state_pass_fraction=0., normalized_error_p99=2.)
    summary = dict(tolerances=[primary], sspi=dict(value=.5), magnitude_bins=[], per_species=[])
    for seed in (20261011, 20261012):
        for name in NAMES:
            directory = tmp_path / f"seed-{seed}" / name
            directory.mkdir(parents=True)
            result = dict(status="complete", config=configuration(name, seed),
                hashes=dict(dataset_manifest="dataset", audit_summary="audit"),
                training=summary, validation=summary, audited_subset=summary, history=[],
                fit_process_seconds=1., total_training_process_seconds=2., total_parameter_count=10,
                inference=dict(median_process_seconds=.01),
                validation_transitions=dict(fail_to_pass=2, pass_to_fail=1),
                validation_physical=dict(negative_endpoint_fraction=.1, inverse_domain_correction_fraction=0.))
            write(directory / "result.json", result)
            check = dict(status="verified", result_sha256=sha256(directory / "result.json"), model_replay="exact")
            check.update({key: True for key in ("independent_counts", "physical_checks", "training_only_preprocessing", "frozen_inputs_unchanged", "transitions_checked")})
            write(directory / "verification.json", check)
    return tmp_path


def test_complete_campaign_keeps_old_evidence(evidence):
    rows, files, identity = collect(evidence)
    assert len(rows["improve_models"]) == 20 and len(files) == 40
    snapshot = dict(id="stable", queries=dict(old=dict(rows=[1]),
        offline_contract=dict(rows=[dict(datasetHash="dataset", auditHash="audit")])))
    original = copy.deepcopy(snapshot)
    updated = extend(snapshot, rows, files, identity)
    assert updated["id"] == original["id"] and updated["queries"]["old"] == original["queries"]["old"]


def test_failure_is_not_a_zero_score(evidence):
    file = evidence / "seed-20261011/local-state/result.json"
    result = json.loads(file.read_text())
    result.update(status="failed", error="Synthetic failure")
    write(file, result)
    rows, _, _ = collect(evidence)
    failed = [row for row in rows["improve_models"] if row["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["componentRate"] is None


def test_changed_evidence_is_rejected(evidence):
    file = evidence / "seed-20261012/finetune-tail/result.json"
    result = json.loads(file.read_text())
    result["fit_process_seconds"] = 0.
    write(file, result)
    with pytest.raises(ValueError, match="verification"):
        collect(evidence)


def test_incomplete_campaign_is_rejected(evidence):
    (evidence / "seed-20261012/local-asinh/verification.json").unlink()
    with pytest.raises(FileNotFoundError):
        collect(evidence)


def test_physics_control_export_requires_bound_verification(tmp_path):
    directory = tmp_path / "physics-prior"
    directory.mkdir()
    tolerance = dict(atol=1e-15, rtol=.1, component_pass_fraction=.8, state_pass_fraction=.2,
                     qualified_pass_fraction=.7, qualified_state_pass_fraction=.1)
    physical = dict(negative_endpoint_fraction=.01, mass_increment_drift=dict(p99=1e-8),
                    heat_release_error_rms_W_m3=2., heat_release_reference_rms_W_m3=4.)
    models = [dict(name=name, validation=dict(tolerances=[tolerance]), audited_subset=dict(tolerances=[tolerance]),
                   inference_cpu_ms_per_state=.2, validation_physical=physical)
              for name in ("rate-euler", "frozen-exponential")]
    write(directory / "summary.json", dict(status="complete", dataset_manifest_sha256="d", audit_sha256="a", models=models))
    write(directory / "verification.json", dict(status="verified", summary_sha256=sha256(directory / "summary.json"),
          prediction_replay="exact", independent_counts=True, physical_checks=True))
    tables, files = physics_tables(tmp_path, ("d", "a"))
    assert len(files) == 2 and tables["physics_models"][0]["heatRelativeRms"] == .5
    with pytest.raises(ValueError, match="identities"):
        physics_tables(tmp_path, ("other", "a"))
    write(directory / "verification.json", dict(status="unverified"))
    with pytest.raises(ValueError, match="mismatch"):
        physics_tables(tmp_path, ("d", "a"))
