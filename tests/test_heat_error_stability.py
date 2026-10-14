"""Heat error must not subtract two large rounded heat-release predictions."""
import numpy as np

from benchmarks.flame_conditioning.metrics import physical_scores


def test_heat_error_accumulates_increment_error_before_enthalpy_contraction():
    states = np.array([[1000., 101325., .5, .5]])
    reference = np.array([[1e-3, -1e-3]])
    predicted = reference.copy()
    predicted[0, 0] = np.nextafter(reference[0, 0], np.inf)
    enthalpy = np.array([1e8, -1e8])
    weights = np.array([2., 28.])
    result = physical_scores(predicted, reference, states, np.zeros_like(predicted, dtype=bool),
        ["A", "B"], np.ones((1, 2)), enthalpy, weights)
    density = states[0, 1]/(8314.46261815324*states[0, 0]*np.sum(states[0, 2:]/weights))
    expected = abs(np.sum((predicted-reference)*enthalpy))*density/1e-6
    np.testing.assert_allclose(result["heat_release_error_rms_W_m3"], expected, rtol=2e-12, atol=0.)
