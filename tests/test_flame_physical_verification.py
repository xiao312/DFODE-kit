import numpy as np
import pytest

ct = pytest.importorskip("cantera")
from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores


def test_independent_density_and_physical_metric_reconciliation():
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1000, 2 * ct.one_atm, "H2:2,O2:1,N2:3.76"
    states = np.array([[gas.T, gas.P, *gas.Y]])
    reference = np.zeros((1, gas.n_species))
    prediction = reference.copy()
    prediction[0, gas.species_index("H")] = 1e-6
    gas.TP = 298.15, ct.one_atm
    recorded = physical_scores(prediction, reference, states, np.zeros_like(reference, dtype=bool),
        gas.species_names, element_matrix(gas),
        gas.standard_enthalpies_RT * ct.gas_constant * gas.T / gas.molecular_weights,
        gas.molecular_weights)
    actual = recompute(states, prediction, reference, "h2o2.yaml", 1e-6)
    assert actual["heat_error_rms"] > 0
    assert_scores(actual, recorded)
    assert actual["temperature_bins"][0]["p99"] is None
    recorded["temperature_bins"][0]["budget_error"]["p99"] = 0
    with pytest.raises(AssertionError, match="empty-bin"):
        assert_scores(actual, recorded)
    recorded["temperature_bins"][0]["budget_error"]["p99"] = None
    recorded["magnitude_bins"][0]["absolute_error"]["count"] += 1
    with pytest.raises(AssertionError, match="identities/counts"):
        assert_scores(actual, recorded)
    recorded["magnitude_bins"][0]["absolute_error"]["count"] -= 1
    recorded["temperature_bins"][3]["budget_error"]["p99"] *= 2
    with pytest.raises(AssertionError, match="temperature_bins p99"):
        assert_scores(actual, recorded)
    recorded["temperature_bins"][3]["budget_error"]["p99"] /= 2
    assert actual["species_budget_p99"]["OH"] == 0.0
    recorded["per_species"]["OH"]["budget_error"]["p99"] = 1.0
    with pytest.raises(AssertionError, match="Saved species p99 differs: OH"):
        assert_scores(actual, recorded)
    recorded["per_species"]["OH"]["budget_error"]["p99"] = 0.0
    recorded["budget_error"]["p99"] *= 2
    with pytest.raises(AssertionError):
        assert_scores(actual, recorded)
