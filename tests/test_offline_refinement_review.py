import copy
import json

import pytest

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.offline_accuracy.refinement.plan import configuration, variants
from benchmarks.offline_accuracy.refinement.review import collect, extend


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    # Synthetic unit-test fixtures only; these are never report evidence.
    primary = dict(atol=1e-15, rtol=.1, states=2, components=4,
                   component_pass_count=1, state_pass_count=0,
                   component_pass_fraction=.25, state_pass_fraction=0.)
    summary = dict(tolerances=[primary], sspi=dict(value=.5), magnitude_bins=[], per_species=[])
    for seed in (20261011, 20261012):
        for variant in variants():
            directory = tmp_path / f"seed-{seed}" / variant["name"]
            directory.mkdir(parents=True)
            result = dict(status="complete", config=configuration(variant["name"], seed),
                hashes=dict(dataset_manifest="dataset", audit_summary="audit"),
                updates_completed=variant["updates"], training_count=10000,
                training=summary, validation=summary, audited_subset=summary,
                total_training_process_seconds=1., total_parameter_count=10,
                inference=dict(median_process_seconds=.01), history=[],
                validation_physical=dict(negative_endpoint_fraction=.1,
                    inverse_domain_correction_fraction=0., mass_increment_drift=dict(p99=1e-8)))
            write(directory / "result.json", result)
            write(directory / "verification-final.json", dict(status="verified", independent_counts=True,
                physical_checks=True, training_only_scales_checked=True,
                model_replay="exact", result_sha256=sha256(directory / "result.json")))
    return tmp_path


def test_retains_all_models_and_old_snapshot(evidence):
    rows, files, identity = collect(evidence)
    assert len(rows["refinement_models"]) == 20
    assert len(files) == 40
    previous = dict(id="stable", queries=dict(old=dict(rows=[dict(user="preserved")]),
        offline_contract=dict(rows=[dict(datasetHash="dataset", auditHash="audit")])))
    result = extend(copy.deepcopy(previous), rows, files, identity)
    assert result["id"] == previous["id"]
    assert result["queries"]["old"] == previous["queries"]["old"]
    assert result["buildStatus"] == "updating"


def test_rejects_changed_result(evidence):
    file = evidence / "seed-20261011/long-state-boxcox/result.json"
    result = json.loads(file.read_text())
    result["validation"]["sspi"]["value"] = 1.
    write(file, result)
    with pytest.raises(ValueError, match="unverified"):
        collect(evidence)


def test_rejects_mixed_dataset_even_with_new_hash(evidence):
    directory = evidence / "seed-20261012/long-state-boxcox"
    result = json.loads((directory / "result.json").read_text())
    result["hashes"]["dataset_manifest"] = "different"
    write(directory / "result.json", result)
    check = json.loads((directory / "verification-final.json").read_text())
    check["result_sha256"] = sha256(directory / "result.json")
    write(directory / "verification-final.json", check)
    with pytest.raises(ValueError, match="differs"):
        collect(evidence)


def test_rejects_incomplete_matrix(evidence):
    file = evidence / "seed-20261012/deep-state-boxcox/verification-final.json"
    file.unlink()
    with pytest.raises(FileNotFoundError):
        collect(evidence)
