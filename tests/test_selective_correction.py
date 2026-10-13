import numpy as np
import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.selective_correction.model import Gate,replace,calibrate


def test_exact_branch_and_endpoint_guard():
    b=np.array([[1e-12,0.]]);c=np.array([[2e-12,-1e-8]])
    out,selected=replace(b,c,np.zeros_like(b),np.ones_like(b),.5)
    np.testing.assert_array_equal(out,[[2e-12,0.]])
    assert selected.tolist()==[[True,False]]


def test_always_base_fallback_and_repair_selection():
    d=np.ones((100,1))*1e-8;b=d.copy();bad=b*2;initial=np.ones_like(b)
    threshold,_=calibrate(np.ones_like(b)*.9,b,bad,initial,d)
    assert threshold==1.01
    b[:10]*=2
    prob=np.r_[np.full((10,1),.9),np.full((90,1),.1)]
    threshold,_=calibrate(prob,b,d,initial,d)
    pred,selected=replace(b,d,initial,prob,threshold)
    assert selected.sum()==10 and np.array_equal(pred,d)


def test_parameter_control_and_no_reference_input():
    a=Gate(119,58,"affine");c=Gate(119,58,"intercept")
    assert sum(p.numel() for p in a.parameters() if p.requires_grad)==7076
    assert sum(p.numel() for p in c.parameters() if p.requires_grad)==58
    assert torch.count_nonzero(a(torch.ones(3,119),torch.ones(3,58)))==0
