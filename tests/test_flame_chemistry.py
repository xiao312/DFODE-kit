import time

import numpy as np
import pytest

ct = pytest.importorskip("cantera")
pytest.importorskip("scipy")

from benchmarks.flame_conditioning.chemistry import (
    EndpointIntegrator, direct_increment, direct_rhs, element_matrix, endpoint, set_state, validate_state,
)


def initial():
    gas = ct.Solution("h2o2.yaml")
    gas.TP = 1400, 101325
    gas.set_equivalence_ratio(1, "H2:1", "O2:1,N2:3.76")
    return gas, {"T": gas.T, "P": gas.P, "Y": gas.Y.tolist()}


def test_state_is_not_silently_normalized():
    gas, state = initial()
    state["Y"] = (np.asarray(state["Y"]) * (1 + 1e-12)).tolist()
    set_state(gas, state)
    np.testing.assert_array_equal(gas.Y, state["Y"])
    state["Y"][0] += 1e-5
    with pytest.raises(ValueError, match="sum to one"):
        set_state(gas, state)


@pytest.mark.parametrize("field,value", [("T", 0), ("P", float("nan")), ("Y", [1])])
def test_invalid_state(field, value):
    gas, state = initial()
    state[field] = value
    with pytest.raises(ValueError):
        validate_state(gas, state)


def test_negative_state_rejected():
    gas, state = initial()
    state["Y"][1] = -1e-30
    with pytest.raises(ValueError):
        validate_state(gas, state)


def test_rhs_conserves_mass_and_elements_at_fixed_temperature_density():
    gas, state = initial()
    scale = np.full(gas.n_species, 1e-6)
    rhs, density = direct_rhs(gas, state, scale)
    rate = rhs(0, np.zeros(gas.n_species)) * scale
    magnitude = max(np.abs(rate).max(), 1)
    assert abs(rate.sum()) < 1e-12 * magnitude
    assert np.abs(element_matrix(gas) @ rate).max() < 1e-12 * magnitude
    assert gas.T == state["T"]
    assert gas.density == pytest.approx(density, rel=1e-14)


def test_cvode_and_direct_agree_but_pressure_can_change():
    _, state = initial()
    result = endpoint("h2o2.yaml", state, interval=1e-6)
    direct = direct_increment("h2o2.yaml", state, interval=1e-6)
    weight = 1e-12 + 1e-6 * np.abs(state["Y"])
    error = np.abs(np.array(result["delta"]) - direct["delta"]) / weight
    assert error.max() < 0.01
    for record in (result, direct):
        diagnostics = record["diagnostics"]
        assert abs(diagnostics["temperature_change_K"]) < 1e-10
        assert abs(diagnostics["relative_density_change"]) < 1e-12
        assert abs(diagnostics["mass_delta_sum"]) < 1e-12
    assert abs(result["diagnostics"]["pressure_change_Pa"]) > 1e-4


def test_direct_deadline():
    _, state = initial()
    with pytest.raises(TimeoutError):
        direct_increment("h2o2.yaml", state, deadline=time.monotonic() - 1)


def test_reused_integrator_matches_fresh_reactor_and_resets_time():
    gas, state = initial()
    reusable = EndpointIntegrator("h2o2.yaml")
    for temperature in (1400, 300, 2200, 1400):
        state["T"] = temperature
        fresh = endpoint("h2o2.yaml", state, atol=1e-21)
        reused = reusable.advance(state)
        np.testing.assert_allclose(reused["delta"], fresh["delta"], rtol=1e-9, atol=1e-20)
        assert reused["diagnostics"]["temperature_change_K"] == 0


@pytest.mark.parametrize("kwargs", [{"interval": 0}, {"rtol": -1}, {"max_step": 0}])
def test_invalid_solve(kwargs):
    _, state = initial()
    with pytest.raises(ValueError):
        endpoint("h2o2.yaml", state, **kwargs)
