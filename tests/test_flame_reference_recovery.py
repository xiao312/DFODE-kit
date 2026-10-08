import numpy as np
import pytest

from benchmarks.flame_conditioning.recover_test_reference import constraints_pass, recovery_decision


def diagnostics(minimum=-1e-36):
    return {"minimum_mass_fraction": minimum, "mass_delta_sum": 0., "element_delta_max": 0.,
            "temperature_change_K": 0., "relative_density_change": 0.}


def checks(delta):
    return {name: {"delta": delta.copy(), "diagnostics": diagnostics()} for name in
            ("fresh", "tighter_absolute", "direct", "direct_tight")}


def test_sign_noise_is_checked_without_clipping_any_input_or_label():
    initial, delta = np.array([1., 0.]), np.array([0., -1e-36])
    saved = delta.copy()
    assert not constraints_pass(diagnostics(), 0)  # Original strict rejection.
    passed, uncertainty = recovery_decision(initial, delta, diagnostics(), checks(delta))
    assert passed and uncertainty == 0
    np.testing.assert_array_equal(delta, saved)
    assert delta[1] < 0


def test_recovery_rejects_material_negativity_disagreement_or_missing_checks():
    initial, delta = np.array([1., 0.]), np.array([0., -1e-36])
    assert not recovery_decision(initial, delta, diagnostics(-1e-20), checks(delta))[0]
    changed = checks(delta)
    changed["direct_tight"]["delta"][1] += 2e-14
    assert not recovery_decision(initial, delta, diagnostics(), changed)[0]
    changed = checks(delta)
    changed["fresh"]["diagnostics"]["relative_density_change"] = 1e-5
    assert not recovery_decision(initial, delta, diagnostics(), changed)[0]
    with pytest.raises(ValueError, match="Every independent"):
        recovery_decision(initial, delta, diagnostics(), {"fresh": changed["fresh"]})
