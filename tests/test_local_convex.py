import numpy as np
import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.local_function.convex import irls_step,objective


def test_majorizer_decreases_same_objective_and_gradient():
    rng=np.random.default_rng(12)
    design=torch.tensor(rng.normal(size=(3,80,5)),dtype=torch.float64)
    truth=torch.tensor(rng.normal(size=(3,5)),dtype=torch.float64)
    target=torch.bmm(design,truth[:,:,None]).squeeze(2)
    target[:,0]+=10
    c=torch.zeros_like(truth)
    before=float(objective(design,target,c))
    for _ in range(20):
        candidate=irls_step(design,target,c)
        after=float(objective(design,target,candidate))
        assert after<=before+1e-12
        c=candidate;before=after
    assert torch.max(abs(c-truth))<.03


def test_rank_deficient_design_remains_regularized():
    design=torch.ones(2,32,4,dtype=torch.float64)/4
    target=torch.ones(2,32,dtype=torch.float64)*.04
    c=irls_step(design,target,torch.zeros(2,4,dtype=torch.float64))
    assert torch.isfinite(c).all()
    np.testing.assert_allclose(c.numpy(),.04,atol=1e-7)
