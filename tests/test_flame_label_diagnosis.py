import numpy as np

from benchmarks.flame_conditioning.diagnose_test_labels import selected_failures, rejection_profile


def test_selection_retains_first_worst_and_temperature_range_without_scores():
    states = np.zeros((20, 3))
    states[:, 0] = np.arange(20) * 100 + 300
    accepted = np.zeros(20, dtype=bool)
    records = [{"diagnostics": {"minimum_mass_fraction": -1e-40}} for _ in states]
    records[7]["diagnostics"]["minimum_mass_fraction"] = -1e-36
    selected = selected_failures(states, records, accepted)
    assert {0, 7, 19} <= set(selected)
    assert len(selected) <= 8
    np.testing.assert_array_equal(selected, selected_failures(states, records, accepted))


def test_profile_distinguishes_tiny_negativity_from_other_failures():
    states = np.array([[300, 101325, 1], [1000, 101325, 1]])
    normal = {"minimum_mass_fraction": -1e-40, "mass_delta_sum": 0,
              "element_delta_max": 0, "temperature_change_K": 0, "relative_density_change": 0}
    records = [{"diagnostics": normal}, {"diagnostics": {**normal, "temperature_change_K": 1}}]
    profile = rejection_profile(states, records, np.array([False, False]), 1e-21)
    assert profile["rejected"] == profile["negative_endpoint_rows"] == 2
    assert profile["other_constraint_failures"] == 1
    assert np.isclose(profile["negative_magnitude_over_solver_atol_max"], 1e-19, rtol=1e-12, atol=0)
    assert sum(row["rejected"] for row in profile["temperature_bins"]) == 2
