import numpy as np
import pytest
import torch

from dfode_kit.physics.thermochemical_closure import (
    Nasa7ThermoData,
    NonPhysicalStateError,
    NumpyThermochemicalClosure,
    SpeciesOnlyClosureContract,
    TemperatureBracketError,
    TorchThermochemicalClosure,
)


def _analytic_thermo_data():
    # Constant-cp NASA7 species. With W_A=2 and W_B=4, 2 A -> B is
    # mass balanced. Different a1 values exercise composition-dependent h.
    low = np.array(
        [
            [3.5, 0.0, 0.0, 0.0, 0.0, 120.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, -80.0, 0.0],
        ],
        dtype=np.float64,
    )
    high = low.copy()
    return Nasa7ThermoData(
        species_names=("A", "B"),
        molecular_weights=np.array([2.0, 4.0]),
        temperature_ranges=np.array([[200.0, 1000.0, 5000.0], [200.0, 1000.0, 5000.0]]),
        low_coefficients=low,
        high_coefficients=high,
    )


def test_numpy_reaction_endpoint_has_float64_thermochemical_closure():
    thermo = NumpyThermochemicalClosure(_analytic_thermo_data())
    y_now = np.array([[0.8, 0.2], [0.675, 0.325]], dtype=np.float64)
    xi = np.array([[0.0625], [0.03125]], dtype=np.float32)
    stoich = np.array([[-2.0], [1.0]], dtype=np.float64)
    expected_y = np.array([[0.55, 0.45], [0.55, 0.45]], dtype=np.float64)
    expected_temperature = np.array([1250.0, 1850.0], dtype=np.float64)
    pressure = np.array([101325.0, 202650.0], dtype=np.float64)
    target_h = thermo.mixture_enthalpy(expected_temperature, expected_y)

    endpoint = thermo.close_reaction_endpoint(y_now, xi, stoich, target_h, pressure)

    assert endpoint.mass_fractions.dtype == np.float64
    assert endpoint.temperature.dtype == np.float64
    assert np.allclose(endpoint.mass_fractions, expected_y, rtol=0.0, atol=1e-8)
    assert np.allclose(endpoint.temperature, expected_temperature, rtol=0.0, atol=1e-10)
    assert np.all(endpoint.temperature_solve.converged)
    assert np.max(np.abs(endpoint.enthalpy - target_h)) < 1e-7
    expected_weight = 1.0 / np.sum(expected_y / np.array([2.0, 4.0]), axis=-1)
    expected_density = pressure * expected_weight / (
        thermo.data.gas_constant * expected_temperature
    )
    assert np.allclose(endpoint.mixture_molecular_weight, expected_weight)
    assert np.allclose(endpoint.density, expected_density)
    assert np.allclose(np.sum(endpoint.mass_fractions, axis=-1), 1.0, atol=1e-15)


def test_numpy_heat_release_and_enthalpy_increment_helpers_are_consistent():
    thermo = NumpyThermochemicalClosure(_analytic_thermo_data())
    y_now = np.array([[0.8, 0.2]], dtype=np.float64)
    y_next = np.array([[0.6, 0.4]], dtype=np.float64)
    temperature = np.array([900.0], dtype=np.float64)
    pressure = np.array([150000.0], dtype=np.float64)
    dt = np.array([1e-6], dtype=np.float64)

    target = thermo.enthalpy_target_from_increment(temperature, y_now, 2500.0)
    assert np.allclose(target - thermo.mixture_enthalpy(temperature, y_now), 2500.0)
    increment = thermo.endpoint_enthalpy_increment(temperature, y_now, temperature, y_next)
    expected_rate = -increment / dt
    specific_rate = thermo.specific_chemical_heat_release_rate(
        temperature, y_now, y_next, dt
    )
    volumetric_rate = thermo.volumetric_chemical_heat_release_rate(
        temperature, pressure, y_now, y_next, dt
    )

    assert np.allclose(specific_rate, expected_rate)
    assert np.allclose(
        volumetric_rate,
        thermo.density(temperature, pressure, y_now) * specific_rate,
    )


def test_numpy_closure_reports_out_of_bracket_and_nonphysical_states():
    thermo = NumpyThermochemicalClosure(_analytic_thermo_data())
    y = np.array([[0.6, 0.4]], dtype=np.float64)
    target_above_bracket = thermo.mixture_enthalpy(np.array([5000.0]), y) + 1e6

    with pytest.raises(TemperatureBracketError):
        thermo.close_endpoint(y, target_above_bracket, 101325.0)

    clamped = thermo.close_endpoint(
        y,
        target_above_bracket,
        101325.0,
        out_of_bracket="clamp",
    )
    assert np.allclose(clamped.temperature, 5000.0)
    assert not bool(np.asarray(clamped.temperature_solve.bracketed).item())
    assert not bool(np.asarray(clamped.temperature_solve.converged).item())

    with pytest.raises(NonPhysicalStateError):
        thermo.close_endpoint(np.array([[1.01, -0.01]]), target_above_bracket, 101325.0)
    with pytest.raises(NonPhysicalStateError):
        thermo.close_endpoint(y, target_above_bracket, -1.0, out_of_bracket="clamp")


def test_torch_closure_is_float64_differentiable_and_traceable():
    thermo = TorchThermochemicalClosure(_analytic_thermo_data())
    y_now = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    xi = torch.tensor([[0.05]], dtype=torch.float32)
    stoich = torch.tensor([[-2.0], [1.0]], dtype=torch.float64)
    y_next = thermo.reaction_endpoint(y_now, xi, stoich)
    expected_temperature = torch.tensor([1400.0], dtype=torch.float64)
    target_h = thermo.mixture_enthalpy(expected_temperature, y_next).detach().requires_grad_()
    pressure = torch.tensor([101325.0], dtype=torch.float64)

    endpoint = thermo.close_endpoint(y_next, target_h, pressure)
    endpoint.temperature.sum().backward()

    assert endpoint.mass_fractions.dtype == torch.float64
    assert endpoint.temperature.dtype == torch.float64
    assert torch.allclose(endpoint.temperature, expected_temperature, rtol=0.0, atol=1e-9)
    assert target_h.grad is not None
    assert torch.all(torch.isfinite(target_h.grad))
    assert torch.all(target_h.grad > 0.0)

    traced = torch.jit.trace(thermo, (y_next, target_h.detach(), pressure))
    traced_temperature, traced_h, traced_density, traced_weight, traced_bracketed = traced(
        y_next, target_h.detach(), pressure
    )
    assert torch.allclose(traced_temperature, endpoint.temperature)
    assert torch.allclose(traced_h, endpoint.enthalpy)
    assert torch.allclose(traced_density, endpoint.density)
    assert torch.allclose(traced_weight, endpoint.mixture_molecular_weight)
    assert bool(torch.all(traced_bracketed))


def test_nasa7_mapping_round_trip_is_explicit_and_lossless():
    data = _analytic_thermo_data()
    restored = Nasa7ThermoData.from_mapping(data.to_mapping())

    assert restored.species_names == data.species_names
    assert np.array_equal(restored.molecular_weights, data.molecular_weights)
    assert np.array_equal(restored.temperature_ranges, data.temperature_ranges)
    assert np.array_equal(restored.low_coefficients, data.low_coefficients)
    assert np.array_equal(restored.high_coefficients, data.high_coefficients)


def test_species_only_contract_preserves_host_temperature_fallback_semantics():
    contract = SpeciesOnlyClosureContract().to_mapping()

    assert contract["predicts_temperature"] is False
    assert contract["temperature_output"] == "none"
    assert contract["temperature_delta_scale"] is None
    assert contract["thermochemical_closure"]["target_enthalpy_source"] == (
        "host_pre_species_replacement"
    )
    assert contract["thermochemical_closure"]["temperature_recovery"] == (
        "host_post_species_replacement"
    )
    assert contract["thermochemical_closure"]["exact_host_equivalence"] is False


def test_offline_temperature_comparison_is_explicitly_non_equivalent_to_host():
    thermo = NumpyThermochemicalClosure(_analytic_thermo_data())
    y = np.array([[0.6, 0.4]], dtype=np.float64)
    offline_temperature = np.array([1200.0], dtype=np.float64)
    host_temperature = np.array([1200.25], dtype=np.float64)
    target_h = thermo.mixture_enthalpy(offline_temperature, y)

    comparison = thermo.compare_offline_to_host_temperature(
        y,
        target_h,
        101325.0,
        host_temperature,
    )

    assert comparison.exact_host_equivalence is False
    assert np.allclose(comparison.temperature_difference, -0.25)
