import numpy as np
import torch
import pytest
from benchmarks.offline_accuracy.paired.coordinates import decode
from benchmarks.offline_accuracy.paper_baseline.paired_base.gpu_model import inverse


def test_gpu_inverse_formula_on_cpu_all_branches():
    initial=np.array([0.,0.,0.,1e-20,1e-8,.1,.1,.1])
    # Include zero initial, forbidden endpoints, small and large changes.
    z=np.array([0.,-1.,1.,1e-9,1.,-1e5,-1.,1e3])
    expected,mask=decode(initial,z,"gbct")
    actual,flags=inverse(torch.tensor(initial),torch.tensor(z))
    np.testing.assert_allclose(actual.numpy(),expected,rtol=2e-13,atol=1e-30)
    np.testing.assert_array_equal(flags.numpy(),mask)


def test_stacked_inverse_matches_scalar_formula_and_is_finite():
    initial=np.logspace(-30,-1,100)
    z=np.stack((np.logspace(-8,2,100),-np.logspace(-8,2,100)))
    value,mask=inverse(torch.tensor(initial)[None],torch.tensor(z))
    for i in range(2):
        reference,flags=decode(initial,z[i],"gbct")
        np.testing.assert_allclose(value[i].numpy(),reference,rtol=3e-13,atol=1e-30)
        np.testing.assert_array_equal(mask[i].numpy(),flags)
    assert np.isfinite(value.numpy()).all()


@pytest.mark.parametrize("initial,z",[(-1.,0.),(float("nan"),0.),(float("inf"),0.),(0.,float("nan")),(.1,float("inf")),(.1,-float("inf")),(.1,-1e200)])
def test_invalid_inverse_inputs_fail_closed(initial,z):
    with pytest.raises(ValueError):
        inverse(torch.tensor([initial],dtype=torch.float64),torch.tensor([z],dtype=torch.float64))
