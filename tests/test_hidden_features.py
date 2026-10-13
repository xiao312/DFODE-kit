import numpy as np
import pytest
torch = pytest.importorskip("torch")
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.paper_baseline.hidden_features.model import ARMS, config
from benchmarks.offline_accuracy.paper_baseline.correction_scaling.objectives import objective


@pytest.mark.parametrize("arm", list(ARMS))
def test_declared_parameter_counts_and_zero_output(arm):
    inputs, widths, count = ARMS[arm]
    model = network(inputs, 58, widths, 20261011, torch.float32, "gelu")
    assert sum(p.numel() for p in model.parameters()) == count
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.zero_()
    assert torch.count_nonzero(model(torch.ones(2, inputs))).item() == 0


def test_fixed_loss_and_no_label_needed_by_inference():
    import inspect
    from benchmarks.offline_accuracy.paper_baseline.hidden_features.model import predict
    assert list(inspect.signature(predict).parameters) == ["base", "head", "prep", "scales", "states", "cfg"]
    cfg = config(20261011, "B")
    assert cfg["updates"] == 6000 and cfg["batch_size"] == 10000
    assert cfg["fit_objective"] == "coordinate-sl1"
    assert cfg["coordinate_beta"] == np.log1p(.1)
    x = torch.zeros((2,58), requires_grad=True)
    target = torch.ones_like(x).double()
    loss = objective(x, target, target, torch.ones_like(target,dtype=torch.bool), target, cfg)
    loss.backward()
    assert torch.isfinite(x.grad).all()


def test_preservation_penalty_allows_interval_and_penalizes_escape():
    from benchmarks.offline_accuracy.paper_baseline.hidden_features.preservation import objective as constrained
    cfg = config(20261011, "P")
    truth = torch.tensor([[1e-12, -1e-12]], dtype=torch.float64)
    base = truth.clone()
    mask = torch.ones_like(base, dtype=torch.bool)
    target = torch.zeros_like(base)
    inside = torch.full_like(base, .01, requires_grad=True)
    assert constrained(inside,base,truth,mask,target,cfg) == objective(inside,base,truth,mask,target,cfg)
    outside = torch.full_like(base, .5, requires_grad=True)
    loss = constrained(outside,base,truth,mask,target,cfg)
    assert loss > objective(outside,base,truth,mask,target,cfg)
    loss.backward()
    assert (outside.grad > 0).all()
