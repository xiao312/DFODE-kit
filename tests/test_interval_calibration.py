import numpy as np
import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.interval_calibration.model import interval_midpoint,smooth_constant,train_constants,calibrated_prediction


def test_smooth_constant_root_is_not_the_mean():
    target=torch.tensor([[0.],[0.],[0.],[20.]],dtype=torch.float64)
    c=smooth_constant(target)
    np.testing.assert_allclose(c.numpy(),np.log1p(.1)/3,atol=1e-15)


def test_smooth_constant_uses_nearest_zero_flat_minimum():
    target=torch.tensor([[-1.,-3.,1.],[1.,-1.,3.]],dtype=torch.float64)
    c=smooth_constant(target).numpy()
    np.testing.assert_allclose(c,[0.,-1.-np.log1p(.1),1.+np.log1p(.1)],atol=1e-14)


def test_interval_consensus_known_overlap_and_tie():
    low=torch.tensor([[-.1,1.],[-.05,2.],[4.,3.]],dtype=torch.float64)
    high=torch.tensor([[.1,1.2],[.05,2.2],[5.,3.2]],dtype=torch.float64)
    c,n=interval_midpoint(low,high)
    np.testing.assert_allclose(c.numpy(),[0.,1.1]);assert n.tolist()==[2,1]
    assert (((c>=low)&(c<=high)).sum(0)==n).all()


def test_interval_learner_cannot_reduce_training_acceptance():
    rng=np.random.default_rng(17)
    d=rng.normal(size=(64,3))*1e-11;b=d*(1+rng.normal(size=(64,3))*.2)
    rows=dict(delta=d,base=b,active=np.ones(3,dtype=bool),target=np.arcsinh(d/1e-14)-np.arcsinh(b/1e-14))
    c,pred,coordinate,detail=train_constants(rows,"interval",device="cpu")
    tau=1e-15+.1*abs(d)
    assert ((abs(pred-d)<=tau).sum(0)>=(abs(b-d)<=tau).sum(0)).all()
    again,ac=calibrated_prediction(b,c,rows["active"])
    np.testing.assert_array_equal(again,pred);np.testing.assert_array_equal(ac,coordinate)


def test_zero_selected_if_base_already_perfect():
    d=np.zeros((6,2));rows=dict(delta=d,base=d.copy(),active=np.ones(2,dtype=bool),target=d.copy())
    c,pred,_,detail=train_constants(rows,"interval",device="cpu")
    assert np.count_nonzero(c)==0 and np.count_nonzero(pred)==0
    assert all(detail["zero_preferred_or_fallback"])
