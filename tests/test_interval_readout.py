import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.interval_readout.targets import targets
from benchmarks.offline_accuracy.paired.coordinates import decode


def test_finite_scale_unbounded_set_and_membership():
    y=np.array([[0.,1e-3,1e-16],[1e-3,1e-4,1e-5]])
    d=np.array([[0.,1e-7,0.],[-1e-7,1e-8,-1e-8]])
    raw=np.zeros_like(d);prep=dict(y_offset=np.zeros(3),y_scale=np.ones(3))
    t,alpha,details=targets(y,d,raw,prep)
    assert np.isfinite(alpha).all() and (alpha>0).all()
    assert np.isneginf(t[0,0,0]) and details['unbounded_lower_components']==2
    for fraction in (.1,.5,.9):
        lo=t[...,0];hi=t[...,1];z=np.where(np.isfinite(lo),(1-fraction)*lo+fraction*hi,hi-1)
        pred,_=decode(y,z*alpha,'gbct')
        assert (abs(pred-d)<=1e-15+.1*abs(d)).all()


def test_interval_zero_gradient_inside_and_point_nonzero():
    torch=pytest.importorskip('torch')
    from benchmarks.offline_accuracy.paper_baseline.interval_readout.targets import objective
    c=torch.tensor([[0.]],requires_grad=True)
    target=torch.tensor([[[-1.,1.,1.,.5]]],dtype=torch.float64)
    mask=torch.ones((1,1),dtype=torch.bool)
    loss=objective(c,None,None,mask,target,dict(arm='interval'));loss.backward()
    assert loss.item()==0 and c.grad.item()==0
    c.grad=None
    point=objective(c,None,None,mask,target,dict(arm='point'));point.backward()
    assert point.item()>0 and c.grad.item()!=0


def test_saturated_lower_no_nan_gradient():
    torch=pytest.importorskip('torch')
    from benchmarks.offline_accuracy.paper_baseline.interval_readout.targets import objective
    c=torch.tensor([[-2.]],requires_grad=True)
    t=torch.tensor([[[float('-inf'),1.,1.,0.]]],dtype=torch.float64)
    value=objective(c,None,None,torch.ones((1,1),dtype=torch.bool),t,dict(arm='interval'));value.backward()
    assert value.item()==0 and c.grad.item()==0
