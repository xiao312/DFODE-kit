import numpy as np

from benchmarks.flame_conditioning.metrics import physical_scores


def evaluate(prediction, corrections=None):
    reference = np.array([[-.01, .01, 0], [-1e-20, 1e-20, 0]])
    states = np.array([[1200, 101325, .2, .7, .1], [300, 101325, .2, .7, .1]])
    return physical_scores(prediction, reference, states,
        np.zeros_like(reference, dtype=bool) if corrections is None else corrections,
        ["A", "B", "AR"], np.ones((1, 3)), np.array([0, -100000, 0]), np.array([2, 4, 40]))


def test_perfect_predictions_and_exact_denominators():
    result = evaluate(np.array([[-.01, .01, 0], [-1e-20, 1e-20, 0]]))
    assert result["samples"] == 2
    assert result["non_argon_species_components"] == 4
    assert result["absolute_error"]["max"] == 0
    assert result["heat_release_error_rms_W_m3"] == 0
    assert result["small_target_count"] == 2
    assert result["small_target_stays_small_fraction"] == 1
    assert sum(row["samples"] for row in result["temperature_bins"]) == 2
    assert sum(row["absolute_error"]["count"] for row in result["magnitude_bins"]) == 4
    assert set(result["per_species"]) == {"A", "B"}


def test_no_hidden_positivity_or_conservation_repair():
    predicted = np.array([[-.3, .1, 0], [0, 0, 0]])
    result = evaluate(predicted, np.array([[True, False, False], [False, False, False]]))
    assert result["negative_endpoint_fraction"] == .25
    assert result["negative_endpoint_row_fraction"] == .5
    assert result["inverse_domain_correction_fraction"] == .25
    assert result["mass_increment_drift"]["max"] > .19
