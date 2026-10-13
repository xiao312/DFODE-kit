import copy
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.research_contract.adapter import aligned_score, project_path
from benchmarks.research_contract.gates import comparison, decision
from benchmarks.research_contract.admission import validate, acquire_lease


ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / "research/campaign.json").read_text())["screening"]


def fixture():
    delta = np.tile([1e-16, 1e-13, 1e-6, 0.], (150, 1))
    base = delta.copy()
    base[:30, :3] *= 2
    rows = dict(delta=delta, base=base, source_indices=np.arange(150),
                states=np.column_stack([np.full(150, 900.), np.full(150, 1e5), np.ones_like(delta)]),
                snapshot=np.full(150, "source"))
    return rows


def score(prediction=None, rows=None):
    rows = fixture() if rows is None else rows
    return aligned_score(dict(prediction=rows["delta"] if prediction is None else prediction,
                              source_indices=rows["source_indices"]), rows, ["A", "B", "C", "AR"],
                         prediction_species=["A", "B", "C", "AR"], population_id="T", prediction_population_id="T")


def test_permuted_rows_rejected():
    rows = fixture()
    with pytest.raises(ValueError, match="Ordered row"):
        aligned_score(dict(prediction=rows["delta"], source_indices=rows["source_indices"][::-1]),
                      rows, ["A", "B", "C", "AR"], prediction_species=["A", "B", "C", "AR"],
                      population_id="T", prediction_population_id="T")


def test_split_identity_rejected():
    rows = fixture()
    with pytest.raises(ValueError, match="Population"):
        aligned_score(dict(prediction=rows["delta"], source_indices=rows["source_indices"]),
                      rows, ["A", "B", "C", "AR"], prediction_species=["A", "B", "C", "AR"],
                      population_id="T", prediction_population_id="D")


def test_exact_candidate_passes_numerical_gates():
    rows = fixture()
    assert comparison(score(), score(rows["base"]), POLICY)["passed"]


def test_loss_is_not_promotion_input():
    rows = fixture()
    bad = score(np.zeros_like(rows["delta"]))
    bad["training_loss"] = 0.
    gate = comparison(bad, score(rows["base"]), POLICY)
    assert "overall_gain" in gate["failures"]
    assert "nontrivial_gain" in gate["failures"]


def test_missing_latency_never_promotes():
    gate = comparison(score(), score(fixture()["base"]), POLICY)
    assert decision({"x":gate}, ["x"], None, POLICY)["candidate_status"] == "INCONCLUSIVE"


def test_both_comparators_required():
    gate = comparison(score(), score(fixture()["base"]), POLICY)
    assert decision({"anchor":gate}, ["anchor", "incumbent"], 1., POLICY)["candidate_status"] == "INCONCLUSIVE"


def test_latency_tradeoff():
    gate = comparison(score(), score(fixture()["base"]), POLICY)
    assert decision({"x":gate}, ["x"], 2., POLICY)["candidate_status"] == "TRADEOFF_ONLY"


def test_zero_baseline_tail_cannot_be_relaxed():
    rows = fixture()
    rows["base"] = rows["delta"].copy()
    baseline = score(rows["base"], rows)
    bad = copy.deepcopy(baseline)
    bad["groups"]["weak_nontrivial"]["normalized_error"]["p99"] = .00001
    assert "weak_nontrivial_p99" in comparison(bad, baseline, POLICY)["failures"]


def test_nonfinite_in_argon_invalidates():
    pred = fixture()["delta"].copy()
    pred[0,-1] = np.nan
    gate = comparison(score(pred), score(fixture()["base"]), POLICY)
    assert "nonfinite_prediction" in gate["failures"]
    assert "argon_invariant" in gate["failures"]


def test_unknown_sparse_group_is_not_success():
    rows = fixture()
    rows = {k:v[:1] for k,v in rows.items()}
    gate = comparison(score(rows=rows), score(rows["base"], rows), POLICY)
    assert "weak_nontrivial_support" in gate["missing"]


def test_comparator_mismatch_rejected():
    baseline = score(fixture()["base"])
    baseline["groups"]["all"]["pass_count"] += 1
    with pytest.raises(ValueError, match="Comparator"):
        comparison(score(), baseline, POLICY)


def test_project_escape_rejected(tmp_path):
    with pytest.raises(ValueError):
        project_path(tmp_path, "../private")


def test_duplicate_launch_lease_rejected(tmp_path):
    path = tmp_path / "lease.json"
    acquire_lease(path, dict(run_id="existing"))
    with pytest.raises(FileExistsError):
        acquire_lease(path, dict(run_id="duplicate"))


def test_admission_checks_authority_and_live_jobs():
    campaign = dict(execution_authorized=False, metric={"atol":1e-15})
    ready = dict(metric=campaign["metric"], m0_verified=True, identities_verified=True,
                 inference_isolation_verified=True, source_clean=True, idle_gpu=True)
    experiment = dict(budget=dict(updates=6000, batch_size=10000))
    with pytest.raises(ValueError, match="authorization"):
        validate(campaign, experiment, ready, [])
    campaign["execution_authorized"] = True
    with pytest.raises(ValueError, match="active job"):
        validate(campaign, experiment, ready, ["job123"])
    ready["idle_gpu"] = False
    assert validate(campaign, experiment, ready, []) == "WAITING_RESOURCE"
    ready["inference_isolation_verified"] = False
    with pytest.raises(ValueError, match="isolation"):
        validate(campaign, experiment, ready, [])
