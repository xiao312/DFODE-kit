import numpy as np
import pytest

from benchmarks.flame_conditioning.pressure_diagnostic import pressure_copy, checked_label


def test_pressure_copy_changes_only_pressure_and_preserves_source():
    states = np.array([[300., 100842., .2, .8], [1500., 100850., .1, .9]])
    original = states.copy()
    changed = pressure_copy(states, 101414.)
    np.testing.assert_array_equal(states, original)
    np.testing.assert_array_equal(changed[:, [0, 2, 3]], states[:, [0, 2, 3]])
    assert np.all(changed[:, 1] == 101414.)
    with pytest.raises(ValueError):
        pressure_copy(states, 0)


def test_pressure_reference_requires_independent_agreement_without_clipping():
    diagnostic = {"minimum_mass_fraction": -1e-36, "mass_delta_sum": 0., "element_delta_max": 0.,
                  "temperature_change_K": 0., "relative_density_change": 0.}
    checks = {key: {"delta": [0., -1e-36], "diagnostics": diagnostic.copy()}
              for key in ("cvode", "step_limited", "direct", "direct_tight")}
    passed, error, label = checked_label(checks, np.array([1., 0.]))
    assert passed and error == 0 and label[1] < 0
    checks["direct_tight"]["delta"][1] = 2e-14
    assert not checked_label(checks, np.array([1., 0.]))[0]
    with pytest.raises(ValueError):
        checked_label({"cvode": checks["cvode"]}, np.array([1., 0.]))
