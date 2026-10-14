import numpy as np
import torch
from benchmarks.offline_accuracy.paper_baseline.element_projection.model import project


def test_uniform_known_solution_and_already_conservative():
    a=torch.tensor([[1.,1.,1.]],dtype=torch.float64)
    b=torch.tensor([[1.,2.,3.],[1.,-1.,0.]],dtype=torch.float64)
    out,flag=project(b,a,"uniform")
    torch.testing.assert_close(out,torch.tensor([[-1.,0.,1.],[1.,-1.,0.]],dtype=torch.float64),atol=1e-14,rtol=0)
    assert not flag.any()


def test_weighted_minimum_norm_and_conservation():
    a=torch.tensor([[1.,1.,0.],[0.,1.,1.]],dtype=torch.float64)
    b=torch.tensor([[.1,.3,-.1]],dtype=torch.float64)
    out,flag=project(b,a,"amplitude")
    d=np.diag(1e-14+abs(b.numpy()[0]));m=a.numpy()
    expected=b.numpy()[0]-d@d@m.T@np.linalg.solve(m@d@d@m.T,m@b.numpy()[0])
    np.testing.assert_allclose(out.numpy()[0],expected,atol=1e-14,rtol=1e-13)
    assert not flag.any()
    assert (out@a.T).abs().max()<1e-14


def test_explicit_numerical_rank_fallback():
    a=torch.tensor([[1.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    b=torch.tensor([[1.,1e-20,0.]],dtype=torch.float64)
    out,flag=project(b,a,"amplitude")
    assert flag.item();torch.testing.assert_close(out,b,rtol=0,atol=0)


def test_no_hidden_positivity_clip():
    a=torch.ones((1,3),dtype=torch.float64);b=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    out,_=project(b,a,"uniform")
    assert out[0,0]<0 and out[0,1]<0
