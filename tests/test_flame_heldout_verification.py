import json

import numpy as np
import pytest

ct = pytest.importorskip("cantera")
from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify_heldout import expected_names, verify


def test_frozen_comparison_includes_each_fixed_hybrid(tmp_path):
    variants = [{"name": "n2-float32-state-boxcox"}, {"name": "n2-float32-signed-power"}]
    (tmp_path / "summary.json").write_text(json.dumps({"variants": variants}))
    plan = {"models": [{"directory": str(tmp_path), "name": "run"}],
            "historical": [{"kind": kind, "modes": ["source-formula", "stable-adapter"]}
                           for kind in ("state-boxcox", "signed-power")]}
    names = expected_names(plan)
    assert len(names) == 10
    assert "run--n2-float32-fixed-hybrid" in names
    assert "historical--fixed-hybrid--source-formula" in names


def test_reconcile_test_populations_and_reject_modified_scores(tmp_path):
    test, evaluation = tmp_path / "test", tmp_path / "evaluation"
    test.mkdir()
    evaluation.mkdir()
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1000, ct.one_atm, "H2:2,O2:1,N2:3.76"
    gas.write_yaml(str(test / "mechanism.yaml"))
    states = np.array([[gas.T, gas.P, *gas.Y]])
    delta = np.zeros((1, gas.n_species))
    prediction = delta.copy()
    delta[0, gas.species_index("H")] = 1e-7
    np.savez(test / "source-states.npz", states=states, sample_row=[7], uniform=[True], balanced=[True])
    np.savez(test / "labels.npz", delta=delta, accepted=[True])
    (test / "frozen-plan.json").write_text(json.dumps({"models": [], "historical": []}))
    manifest = {"status": "complete", "interval_s": 1e-6}
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("mechanism.yaml", "mechanism_sha256"), ("frozen-plan.json", "frozen_plan_sha256")):
        manifest[key] = sha256(test / name)
    (test / "manifest.json").write_text(json.dumps(manifest))
    gas.TP = 298.15, ct.one_atm
    scores = physical_scores(prediction, delta, states, np.zeros_like(delta, dtype=bool),
        gas.species_names, element_matrix(gas),
        gas.standard_enthalpies_RT * ct.gas_constant * gas.T / gas.molecular_weights, gas.molecular_weights)
    summary = {"status": "complete", "test_manifest_sha256": sha256(test / "manifest.json"),
               "models": [{"name": "zero-baseline", "populations": {"uniform": scores, "balanced": scores}}],
               "sample_counts": {name: {"selected": 1, "accepted": 1} for name in ("uniform", "balanced")}}
    np.savez(evaluation / "zero-baseline.npz", prediction=prediction, cell_ids=[7])
    summary_path = evaluation / "summary.json"
    summary_path.write_text(json.dumps(summary))
    result = verify(test, evaluation)
    assert result["status"] == "verified" and result["population_scores"] == 2
    scores["budget_error"]["p99"] *= 2
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(AssertionError):
        verify(test, evaluation)
