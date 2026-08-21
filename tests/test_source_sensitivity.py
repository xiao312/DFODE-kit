import pytest
import torch

from dfode_kit.models.source_sensitivity import (
    CompactDiagonalSourceDerivativeHead,
    StabilizingDiagonalSourceAdapter,
    endpoint_diagonal_to_source_diagonal,
)


def test_endpoint_diagonal_conversion_and_hard_stabilizing_projection():
    raw = endpoint_diagonal_to_source_diagonal(
        endpoint_diagonal=torch.tensor([[0.5, 1.2]], dtype=torch.float64),
        density=torch.tensor([2.0]),
        dt=torch.tensor([0.1]),
    )
    assert raw == pytest.approx(torch.tensor([[-10.0, 4.0]]))
    adapter = StabilizingDiagonalSourceAdapter(
        8.0, policy="hard-negative"
    )
    constrained = adapter(raw)
    assert constrained == pytest.approx(torch.tensor([[-8.0, 0.0]]))
    assert adapter.diagonal_only is True
    assert adapter.full_jacobian is False


def test_density_derivative_term_is_included_explicitly():
    raw = endpoint_diagonal_to_source_diagonal(
        endpoint_diagonal=torch.ones((1, 2)),
        density=torch.tensor([1.0]),
        dt=torch.tensor([0.5]),
        delta_y=torch.tensor([[0.1, -0.2]]),
        density_diagonal=torch.tensor([[2.0, 3.0]]),
    )
    assert raw == pytest.approx(torch.tensor([[0.4, -1.2]]))


def test_smooth_adapter_is_bounded_negative_and_differentiable():
    raw = torch.tensor([[-20.0, 0.0, 20.0]], requires_grad=True)
    adapter = StabilizingDiagonalSourceAdapter(
        10.0, policy="smooth-negative", smoothness=0.1
    )
    constrained = adapter(raw)
    assert constrained.dtype == torch.float64
    assert torch.all(constrained <= 0.0)
    assert torch.all(constrained >= -10.0)
    constrained.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_compact_head_returns_float64_bounded_diagonal_and_gradients():
    head = CompactDiagonalSourceDerivativeHead(
        input_dim=5,
        n_species=3,
        hidden_dim=12,
        bottleneck_dim=4,
        max_abs_derivative=torch.tensor([10.0, 20.0, 30.0]),
        initial_fraction=1.0e-3,
    )
    features = torch.randn(4, 5, requires_grad=True)
    output = head(features)
    diagonal = output["source_derivative_diagonal"]
    assert diagonal.shape == (4, 3)
    assert diagonal.dtype == torch.float64
    assert torch.all(diagonal <= 0.0)
    assert torch.all(diagonal >= torch.tensor([-10.0, -20.0, -30.0]))
    assert output["raw_logits"].dtype == torch.float32
    assert head.diagonal_only is True
    assert head.full_jacobian is False
    diagonal.sum().backward()
    assert features.grad is not None


def test_adapter_rejects_unknown_policy():
    with pytest.raises(ValueError, match="policy"):
        StabilizingDiagonalSourceAdapter(1.0, policy="full-jacobian")
