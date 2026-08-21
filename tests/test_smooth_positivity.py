import torch

from dfode_kit.physics.smooth_positivity import (
    SmoothReactionExtentLimiter,
    compare_hard_and_smooth_scaling,
    deployment_limiter_diagnostics,
    directional_smoothness_diagnostics,
    hard_patankar_process_availability,
    patankar_directional_smoothness_diagnostics,
    reaction_coordinate_delta,
    smooth_safe_patankar_process_availability,
    smooth_safe_reaction_extent_scaling,
)


def _balanced_stoichiometric_matrix():
    return torch.tensor(
        [
            [-1.0, 0.0],
            [-2.0, -1.0],
            [3.0, -1.0],
            [0.0, 2.0],
        ],
        dtype=torch.float64,
    )


def test_smooth_extent_scaling_is_safe_conserved_and_float64():
    stoich = _balanced_stoichiometric_matrix()
    current = torch.tensor(
        [
            [1e-18, 0.20, 0.30, 0.50 - 1e-18],
            [0.10, 1e-14, 0.45, 0.45 - 1e-14],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    extent = torch.tensor(
        [[0.08, 0.03], [0.02, 0.20], [0.30, -0.10]],
        dtype=torch.float32,
    )
    result = smooth_safe_reaction_extent_scaling(
        current, extent, stoich, numerical_tolerance=1e-15
    )
    assert result.next_y.dtype == torch.float64
    assert torch.all(result.next_y >= -1.01e-15)
    assert torch.all(result.scale <= result.hard_cap + 2e-15)
    assert torch.allclose(
        result.delta_y, result.scaled_extent @ stoich.T, atol=0.0, rtol=0.0
    )
    assert torch.allclose(
        result.delta_y.sum(dim=-1),
        torch.zeros(current.shape[0], dtype=torch.float64),
        atol=2e-15,
        rtol=0.0,
    )


def test_sparse_and_dense_reaction_coordinate_updates_match():
    stoich = _balanced_stoichiometric_matrix()
    extent = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64)
    dense = reaction_coordinate_delta(extent, stoich)
    sparse = reaction_coordinate_delta(extent, stoich.to_sparse())
    assert torch.allclose(dense, sparse, atol=1e-15, rtol=0.0)


def test_smooth_scale_is_a_safe_lower_bound_to_hard_limiter():
    generator = torch.Generator().manual_seed(260624)
    stoich = _balanced_stoichiometric_matrix()
    current = torch.rand(128, 4, generator=generator, dtype=torch.float64)
    current = current / current.sum(dim=-1, keepdim=True)
    extent = 0.4 * torch.randn(128, 2, generator=generator, dtype=torch.float64)
    hard, smooth, diagnostics = compare_hard_and_smooth_scaling(
        current, extent, stoich
    )
    assert torch.all(smooth.scale <= hard.scale + 2e-15)
    assert torch.all(smooth.next_y >= -1e-15)
    assert diagnostics["maximum_smooth_minus_hard_scale"] <= 2e-15


def test_smooth_limiter_has_finite_gradients_at_competing_species():
    stoich = torch.tensor(
        [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=torch.float64
    )
    current = torch.tensor(
        [[0.1, 0.1, 0.8]], dtype=torch.float64, requires_grad=True
    )
    extent = torch.tensor([[0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    result = smooth_safe_reaction_extent_scaling(current, extent, stoich)
    gradient_y, gradient_extent = torch.autograd.grad(
        result.delta_y.square().sum(), (current, extent)
    )
    assert torch.all(torch.isfinite(gradient_y))
    assert torch.all(torch.isfinite(gradient_extent))


def test_smooth_limiter_removes_hard_active_species_jvp_kink():
    stoich = torch.tensor(
        [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]], dtype=torch.float64
    )
    current = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
    extent = torch.tensor([[0.2, 0.2]], dtype=torch.float64)
    direction = torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64)
    diagnostics = directional_smoothness_diagnostics(
        current, extent, stoich, direction, epsilon=1e-7
    )
    assert diagnostics["hard_active_species_switch_rate"] == 1.0
    assert diagnostics["smooth_directional_jvp_jump_l2_mean"] < (
        1e-3 * diagnostics["hard_directional_jvp_jump_l2_mean"]
    )


def test_smooth_patankar_bound_is_safe_and_removes_reactant_switch_kink():
    consumption = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64)
    current = torch.tensor([[0.1, 0.1, 0.8]], dtype=torch.float64)
    demand = torch.tensor([[1.0]], dtype=torch.float64)
    hard = hard_patankar_process_availability(current, demand, consumption)
    smooth = smooth_safe_patankar_process_availability(current, demand, consumption)
    consumed = smooth.process_extent @ consumption.T
    assert torch.all(smooth.process_scale <= hard.process_scale)
    assert torch.all(consumed <= current + 1e-15)
    diagnostics = patankar_directional_smoothness_diagnostics(
        current,
        demand,
        consumption,
        torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float64),
        epsilon=1e-7,
    )
    assert diagnostics["hard_active_reactant_switch_rate"] == 1.0
    assert diagnostics["smooth_process_jvp_jump_l2_mean"] < (
        1e-3 * diagnostics["hard_process_jvp_jump_l2_mean"]
    )


def test_strict_smooth_bound_makes_material_deployment_limiter_redundant():
    stoich = _balanced_stoichiometric_matrix()
    current = torch.tensor(
        [[0.0, 0.2, 0.3, 0.5], [0.1, 1e-18, 0.4, 0.5 - 1e-18]],
        dtype=torch.float64,
    )
    extent = torch.tensor([[0.1, 0.2], [0.2, 0.3]], dtype=torch.float64)
    smooth = smooth_safe_reaction_extent_scaling(
        current, extent, stoich, numerical_tolerance=0.0
    )
    diagnostics = deployment_limiter_diagnostics(current, smooth.delta_y)
    assert torch.all(smooth.next_y >= 0.0)
    assert diagnostics["material_activation_rate"] == 0.0


def test_module_api_returns_reaction_space_outputs():
    stoich = _balanced_stoichiometric_matrix()
    layer = SmoothReactionExtentLimiter(stoich, numerical_tolerance=0.0)
    current = torch.full((2, 4), 0.25, dtype=torch.float64)
    extent = torch.tensor([[0.5, 0.2], [-0.1, 0.3]], dtype=torch.float32)
    output = layer(extent, current)
    assert output["next_y"].shape == current.shape
    assert output["reaction_extent"].shape == extent.shape
    assert torch.all(output["next_y"] >= 0.0)
