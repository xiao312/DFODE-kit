import pytest
import torch

from dfode_kit.training.fluent_deployment_losses import (
    FLUENT_NATIVE_DI_BACKEND,
    FluentDeploymentLossConfig,
    FluentJVPConsistencyConfig,
    LossTerm,
    fluent_deployment_losses,
    fluent_di_jvp_consistency_losses,
    ideal_gas_density,
    mixture_molecular_weight,
)


def _species_enthalpy(temperature):
    return torch.stack([2.0 * temperature, 4.0 * temperature], dim=-1)


def test_deployment_losses_are_float64_and_report_explicit_scaling():
    predicted_delta = torch.tensor([[-0.05, 0.05]], requires_grad=True)
    target_delta = torch.tensor([[-0.10, 0.10]], dtype=torch.float64)
    config = FluentDeploymentLossConfig(
        source=LossTerm(weight=3.0, scale=0.05),
        mixture_molecular_weight=LossTerm(weight=2.0, scale=0.1),
        density_increment=LossTerm(weight=4.0, scale=1.0e-4),
        enthalpy=LossTerm(weight=0.5, scale=100.0),
        heat_release=LossTerm(weight=0.25, scale=10.0),
    )
    result = fluent_deployment_losses(
        current_y=torch.tensor([[0.5, 0.5]]),
        predicted_delta_y=predicted_delta,
        target_delta_y=target_delta,
        current_temperature=torch.tensor([1000.0]),
        predicted_temperature=torch.tensor([1005.0]),
        target_temperature=torch.tensor([1010.0]),
        pressure=torch.tensor([8314.46261815324]),
        dt=torch.tensor([0.5]),
        molecular_weights=torch.tensor([2.0, 4.0]),
        species_enthalpy=_species_enthalpy,
        config=config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )

    source = result["terms"]["source"]
    assert source["raw_mae"].detach().item() == pytest.approx(0.1)
    assert source["scale"] == pytest.approx(0.05)
    assert source["scaled"].detach().item() == pytest.approx(2.0)
    assert source["weight"] == pytest.approx(3.0)
    assert source["contribution"].detach().item() == pytest.approx(6.0)
    assert result["total"].dtype == torch.float64
    assert all(
        term["contribution"].dtype == torch.float64
        for term in result["terms"].values()
    )
    result["total"].backward()
    assert predicted_delta.grad is not None


def test_thermodynamic_helpers_use_unmodified_mass_fractions():
    y = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    weights = torch.tensor([2.0, 4.0], dtype=torch.float64)
    expected_weight = 1.0 / (0.5 / 2.0 + 0.5 / 4.0)
    assert mixture_molecular_weight(y, weights) == pytest.approx(expected_weight)
    density = ideal_gas_density(
        pressure=torch.tensor([8314.46261815324]),
        temperature=torch.tensor([1000.0]),
        mass_fractions=y,
        molecular_weights=weights,
        gas_constant=8314.46261815324,
    )
    assert density == pytest.approx(expected_weight / 1000.0)


def test_directional_endpoint_and_source_jvp_match_fluent_di_pairs():
    epsilon = torch.tensor([1.0e-3], dtype=torch.float64)
    dt = torch.tensor([1.0e-6], dtype=torch.float64)
    current = torch.tensor(
        [[1000.0, 101325.0, 0.8, 0.2]], dtype=torch.float64
    )
    direction = torch.tensor(
        [[10.0, 0.0, -0.1, 0.1]], dtype=torch.float64
    )
    perturbed = current + epsilon[:, None] * direction
    target = current + torch.tensor(
        [[1.0, 0.0, -0.01, 0.01]], dtype=torch.float64
    )
    target_perturbed = target + epsilon[:, None] * (2.0 * direction)
    config = FluentJVPConsistencyConfig(
        endpoint=LossTerm(weight=2.0, scale=10.0),
        source=LossTerm(weight=3.0, scale=1.0e6),
    )
    exact = fluent_di_jvp_consistency_losses(
        current_state=current,
        perturbed_state=perturbed,
        predicted_endpoint=target.clone().requires_grad_(),
        predicted_perturbed_endpoint=target_perturbed.clone().requires_grad_(),
        target_di_endpoint=target,
        target_di_perturbed_endpoint=target_perturbed,
        epsilon=epsilon,
        dt=dt,
        config=config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )
    assert exact["total"].detach().item() == pytest.approx(0.0)
    assert exact["full_jacobian_reconstructed"] is False
    assert exact["derived"]["predicted_source_jvp"].shape == (1, 2)

    biased_endpoint = target_perturbed.clone()
    biased_endpoint[:, 2] += 2.0e-4
    biased = fluent_di_jvp_consistency_losses(
        current_state=current,
        perturbed_state=perturbed,
        predicted_endpoint=target,
        predicted_perturbed_endpoint=biased_endpoint,
        target_di_endpoint=target,
        target_di_perturbed_endpoint=target_perturbed,
        epsilon=epsilon,
        dt=dt,
        config=config,
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )
    assert biased["terms"]["endpoint_jvp"]["raw_mae"] > 0.0
    assert biased["terms"]["source_jvp"]["raw_mae"] > 0.0


@pytest.mark.parametrize("backend", ["cantera", "cvode", "sundials", "unknown"])
def test_losses_reject_non_fluent_native_labels(backend):
    config = FluentJVPConsistencyConfig(
        endpoint=LossTerm(weight=1.0, scale=1.0),
        source=LossTerm(weight=1.0, scale=1.0),
    )
    state = torch.zeros((1, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="Fluent native"):
        fluent_di_jvp_consistency_losses(
            current_state=state,
            perturbed_state=state,
            predicted_endpoint=state,
            predicted_perturbed_endpoint=state,
            target_di_endpoint=state,
            target_di_perturbed_endpoint=state,
            epsilon=torch.ones(1),
            dt=torch.ones(1),
            config=config,
            label_backend=backend,
        )
