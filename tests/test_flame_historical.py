import numpy as np
import pytest

pytest.importorskip("torch")
from benchmarks.flame_conditioning.historical import reconstruct, transformed_inputs


def fixture():
    states = np.zeros((2, 61))
    states[:, :2] = [800, 101325]
    states[:, 2] = .2
    states[:, 60] = .8
    arrays = {"features_mean": np.zeros(61), "features_std": np.ones(61),
              "labels_mean": np.zeros(58), "labels_std": np.ones(58)}
    return states, arrays


def test_historical_inputs_keep_pressure_pa_and_zero_without_floor():
    states, _ = fixture()
    values = transformed_inputs(states, "signed-power")
    assert values[0, 1] == 101325
    assert values[0, 3] == -10
    states[0, 2] = -1e-30
    with pytest.raises(ValueError):
        transformed_inputs(states, "state-boxcox")


def test_power_source_has_no_lambda_factor_and_argon_is_inert():
    states, arrays = fixture()
    output = np.full((2, 58), -.2, dtype=np.float32)
    for mode in ("source-formula", "stable-adapter"):
        delta, correction, violation = reconstruct(states, np.zeros_like(states), output, arrays, "signed-power", mode)
        np.testing.assert_allclose(delta[:, :58], -float(np.float32(.2)) ** 10, rtol=2e-6)
        assert not delta[:, 58].any() and not correction.any() and not violation.any()


def test_boxcox_source_and_stable_modes_expose_domain_differently():
    states, arrays = fixture()
    inputs = transformed_inputs(states, "state-boxcox").astype(np.float32)
    output = np.zeros((2, 58), dtype=np.float32)
    output[:, 1] = -1
    source, repaired_source, source_violation = reconstruct(states, inputs, output, arrays, "state-boxcox", "source-formula")
    stable, repaired_stable, stable_violation = reconstruct(states, inputs, output, arrays, "state-boxcox", "stable-adapter")
    assert (source[:, 1] > 0).all()  # Unphysical negative base raised to even power.
    assert not repaired_source.any() and source_violation[:, 1].all()
    assert not stable[:, 1].any() and repaired_stable[:, 1].all() and stable_violation[:, 1].all()


def test_source_formula_rounding_can_erase_small_increment():
    states, arrays = fixture()
    inputs = transformed_inputs(states, "state-boxcox").astype(np.float32)
    output = np.zeros((2, 58), dtype=np.float32)
    output[:, 0] = 1e-10
    raw, _, _ = reconstruct(states, inputs, output, arrays, "state-boxcox", "source-formula")
    stable, _, _ = reconstruct(states, inputs, output, arrays, "state-boxcox", "stable-adapter")
    assert not raw[:, 0].any() and (stable[:, 0] > 0).all()
