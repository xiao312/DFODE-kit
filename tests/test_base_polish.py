import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.adaptive.fit import backward_mean


@pytest.mark.parametrize("chunk",[1,3,7])
def test_microbatch_preserves_row_mean_gradient(chunk):
    x=torch.arange(14,dtype=torch.float64).reshape(7,2)/10
    w=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True)
    def objective(indices):return (x[indices]@w-1).square().mean()
    expected=objective(torch.arange(7));expected.backward();gradient=w.grad.clone();w.grad=None
    actual=backward_mean(torch.arange(7),objective,chunk)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(w.grad,gradient)
