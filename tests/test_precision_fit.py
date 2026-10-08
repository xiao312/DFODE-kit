"""Training-fit checks use synthetic arrays, never run chemistry."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks/precision_conditioning/learning"))


def module():
    pytest.importorskip("torch")
    pytest.importorskip("cantera")
    import fit_diagnostic
    return fit_diagnostic


def test_physical_loss_and_gradient_match_direct_budget_error():
    fit = module()
    torch = fit.torch
    q = torch.tensor([[.4, .1, .9], [.2, -.3, .2]], dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(q)
    scale = torch.tensor([2., 100., 1.], dtype=torch.float64)
    active = torch.tensor([True, True, False])
    loss = fit.objective(q, target, scale, active, "physical-budget")
    direct = (((q - target)[:, active] * scale[active])**2).mean() / 100**2
    torch.testing.assert_close(loss, direct)
    gradient = torch.autograd.grad(loss, q, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.autograd.grad(direct, q)[0])
    assert torch.all(gradient[:, 2] == 0)
    assert not torch.isclose(loss, fit.objective(q, target, scale, active, "coordinate"))


def test_subsets_reject_held_out_rows_and_require_eight():
    fit = module()
    data = {"id": np.arange(9), "split": np.array(["train"] * 9),
            "parent": np.array(["h2-1400K"] * 8 + ["h2-900K"]), "inputs": np.full((9, 3), -6.)}
    config = {"parent_temperature_K": 1400, "interval_s": 1e-6, "narrow_rows": 8}
    chosen = fit.subsets(data, "h2", config)
    assert {key: len(value) for key, value in chosen.items()} == {"one": 1, "narrow": 8, "all": 9}
    data["split"][0] = "test"
    with pytest.raises(ValueError, match="training rows only"):
        fit.subsets(data, "h2", config)


def test_loader_discards_validation_and_test(tmp_path):
    fit = module()
    path = tmp_path / "dataset.npz"
    np.savez(path, id=np.array(["a", "b", "c"]), split=np.array(["train", "validation", "test"]), delta=np.arange(6).reshape(3, 2))
    loaded = fit.training_data(path)
    assert loaded["id"].tolist() == ["a"]
    np.testing.assert_array_equal(loaded["delta"], [[0, 1]])
