import numpy as np
import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.local_function.model import Basis,geometry,select_centres,cached_coordinate


def test_centres_are_deterministic_common_and_unique():
    ids=np.arange(24)[::-1];snapshots=np.repeat(["a","b","c","d"],6)
    a=select_centres(ids,snapshots,17,8)
    np.testing.assert_array_equal(a,select_centres(ids,snapshots,17,8))
    assert len(np.unique(a))==8
    assert all((snapshots[a]==s).sum()==2 for s in np.unique(snapshots))


def test_distance_direct_formula_and_zero_far_field():
    c=np.array([[1.,2.],[-1.,0.]])
    w=np.array([[1.,1.],[.1,1.9]])
    model=Basis(c,w,np.ones(2))
    x=torch.tensor([[2.,-3.],[0.,0.]],dtype=torch.float32)
    expected=((x.numpy()[:,None,None,:]-c[None,None,:,:])**2*w[None,:,None,:]).sum(axis=3)
    np.testing.assert_allclose(model.distances(x).numpy(),expected,atol=1e-12)
    phi=model(x)
    assert torch.all(phi.sum(2)<1)
    assert torch.count_nonzero(model(torch.ones(1,2)*1e6))==0
    coef=torch.nn.Parameter(torch.zeros(2,2))
    out=torch.einsum("nsk,sk->ns",phi,coef)
    assert torch.count_nonzero(out)==0
    (out-1).square().mean().backward()
    assert torch.isfinite(coef.grad).all() and coef.grad.abs().sum()>0


def test_bandwidth_degeneracy_declared_and_roundtrip():
    x=np.zeros((8,2),dtype=np.float32)
    width,fallback=geometry(x,np.array([0,1]),np.ones((2,2)),device="cpu")
    np.testing.assert_array_equal(width,np.ones(2));assert all(fallback)
    basis=Basis(x[:2],np.ones((2,2)),width)
    coef=torch.ones(2,2)
    np.testing.assert_array_equal(cached_coordinate(basis,coef,x),cached_coordinate(basis,coef,x))
