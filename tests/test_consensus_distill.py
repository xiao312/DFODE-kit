import numpy as np
import torch
from benchmarks.offline_accuracy.paper_baseline.consensus_distill.model import cap,decode,teacher_fraction,objective
from benchmarks.offline_accuracy.paper_baseline.paired_base.metrics import bounded_mean,strict_shrink


def test_teacher_matches_registered_rule_to_arithmetic_resolution():
    b=np.r_[0.,np.logspace(-30,-2,100),-np.logspace(-30,-2,100)]
    for q in (2*b,.5*b,-b):
        y=decode(b,teacher_fraction(b,q));expected=strict_shrink(b,bounded_mean(b,q))
        np.testing.assert_allclose(y,expected,rtol=3e-16,atol=0)
        np.testing.assert_array_equal(np.sign(y),np.sign(b))


def test_decoder_bounds_and_initial_half_fraction():
    b=np.array([-1e-3,-1e-20,0.,1e-20,1e-3])
    for alpha in (0.,.5,1.):
        y=decode(b,np.full_like(b,alpha))
        assert np.all(abs(y)<=abs(b));np.testing.assert_array_equal(np.sign(y),np.sign(b))
        assert np.all(abs(y-b)<=cap(b)*1.0000000001)


def test_soft_bce_has_finite_gradient_without_reference_dependence():
    x=torch.zeros((2,3),requires_grad=True);target=torch.tensor([[0.,.5,1.],[1.,0.,.25]],dtype=torch.float64)
    mask=torch.ones_like(target,dtype=torch.bool)
    loss=objective(x,None,None,mask,target,{})
    loss.backward();torch.testing.assert_close(x.grad,(.5-target).float()/6)


def test_head_parameter_budget():
    model=torch.nn.Sequential(torch.nn.Linear(800,64),torch.nn.GELU(),torch.nn.Linear(64,58))
    assert sum(p.numel() for p in model.parameters())==55034
