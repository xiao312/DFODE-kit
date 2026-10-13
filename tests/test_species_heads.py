import inspect
import numpy as np
import torch
from benchmarks.offline_accuracy.paper_baseline.species_heads.model import SpeciesHeads
from benchmarks.offline_accuracy.paper_baseline.correction_scaling.train import fit
from benchmarks.offline_accuracy.paper_baseline.resolved_fit.fit import head


def test_parameter_count_and_zero_output():
    model=SpeciesHeads(119,58,11)
    assert sum(p.numel() for p in model.parameters())==285882
    torch.testing.assert_close(model(torch.ones(3,119)),torch.zeros(3,58),rtol=0,atol=0)
    assert inspect.signature(fit).parameters['model_factory'].default is head


def test_independent_forward_and_no_cross_species_gradients():
    model=SpeciesHeads(7,3,12)
    with torch.no_grad():model.w3.fill_(.1)
    x=torch.linspace(-1,1,35).reshape(5,7)
    expected=[]
    for s in range(3):
        h=torch.nn.functional.gelu(x@model.w1[s]+model.b1[s])
        h=torch.nn.functional.gelu(h@model.w2[s]+model.b2[s])
        expected.append(h@model.w3[s]+model.b3[s])
    torch.testing.assert_close(model(x),torch.stack(expected,dim=1),rtol=1e-5,atol=1e-7)
    model(x)[:,1].sum().backward()
    for p in model.parameters():
        assert torch.count_nonzero(p.grad[[0,2]])==0
        assert torch.count_nonzero(p.grad[1])>0


def test_deterministic_seed_and_replay():
    a=SpeciesHeads(7,3,17);b=SpeciesHeads(7,3,17)
    for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    with torch.no_grad():a.w3.fill_(.3)
    b.load_state_dict(a.state_dict())
    x=torch.ones(4,7);np.testing.assert_array_equal(a(x).detach().numpy(),b(x).detach().numpy())
