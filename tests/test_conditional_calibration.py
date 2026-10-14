import numpy as np
import pytest
pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.conditional_calibration.model import partition,fit_map,predict,fold_ids


def test_duplicate_quantiles_and_edges():
    edges,group=partition(np.zeros(100),16)
    assert len(edges)==1 and np.all(group==1)
    e,g=partition(np.arange(100)*1e-12,16)
    assert len(e)==15 and len(np.unique(g))==16


def test_fold_exclusion_and_replay():
    ids=fold_ids(100,17)
    assert np.bincount(ids).tolist()==[50,50]
    np.testing.assert_array_equal(ids,fold_ids(100,17))


def test_conditional_bias_is_not_constant_shift():
    b=np.r_[np.full(50,1e-12),np.full(50,1e-8)][:,None]
    d=b*np.r_[np.full(50,1.3),np.full(50,.7)][:,None]
    fitted=fit_map(b,d,2,8,device="cpu")
    y=predict(b,fitted)
    assert np.all(abs(y-d)<=1e-15+.1*abs(d))
    # Query needs neither training nor query references.
    assert predict(b[:2],fitted).shape==(2,1)


def test_sparse_identity_and_species_axis_guard():
    b=np.ones((4,2))*1e-9
    model=fit_map(b,b*2,16,128,device="cpu")
    np.testing.assert_array_equal(predict(b,model),b)
    with pytest.raises(ValueError):predict(b[:,:1],model)
