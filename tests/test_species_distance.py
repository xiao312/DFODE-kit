import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.species_distance.metric import calibration_mask,rank_weights,scale_features
from benchmarks.offline_accuracy.paper_baseline.species_distance.analysis import transfer_species
from benchmarks.offline_accuracy.paper_baseline.local_predictability.analysis import transfer


def test_queries_excluded_and_missing_identity_fails():
    np.testing.assert_array_equal(calibration_mask(np.arange(5),np.array([1,3])),[True,False,True,False,True])
    with pytest.raises(ValueError):
        calibration_mask(np.arange(5),np.array([8]))


def test_rank_weights_floor_ties_and_constant_fallback():
    x = np.array([[0,3,1],[0,2,1],[2,1,1],[3,0,1]],dtype=float)
    y = np.column_stack((x[:,0],np.ones(4)))
    w,rho = rank_weights(x,y)
    np.testing.assert_allclose(rho[0,0],1)
    assert rho[0,2] == 0 and rho[0,1] < 0
    assert (w >= .1).all()
    np.testing.assert_allclose(w.mean(axis=1),1)
    np.testing.assert_array_equal(w[1],np.ones(3))
    np.testing.assert_allclose(rank_weights(x**3,y*10)[0],w)


def test_weights_fit_only_selected_rows():
    x = np.arange(15,dtype=float).reshape(5,3)
    y = np.arange(5,dtype=float)[:,None]
    mask = calibration_mask(np.arange(5),np.array([4]))
    before = rank_weights(x[mask],y[mask])[0]
    y[4] = -1e10
    np.testing.assert_array_equal(rank_weights(x[mask],y[mask])[0],before)


@pytest.mark.parametrize("mode",["target","correction"])
def test_species_gather_and_common_graph_parity(mode):
    truth = np.array([[0,1e-12,-1e-5],[0,2e-10,3e-7]])
    base = np.array([[0,1e-8,1e-5],[0,1e-20,-1e-6]])
    active = np.array([False,True,True])
    library = dict(delta=truth,base=base)
    ix = np.array([[0],[1]])
    common = np.repeat(ix[:,None,:],2,axis=1)
    got,c = transfer_species(library,base,common,active,mode)
    old,old_c = transfer(library,base,ix,active,mode)
    np.testing.assert_array_equal(got,old)
    np.testing.assert_array_equal(c,old_c)
    mixed = np.array([[[0],[1]],[[1],[0]]])
    pred,_ = transfer_species(library,base,mixed,active,"target")
    expected = truth.copy()
    expected[:,2] = truth[::-1,2]
    assert (abs(pred-expected)<=1e-15+.1*abs(expected)).all()


def test_weighted_search_on_cpu_and_uniform_parity():
    pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.local_predictability.neighbors import search,verify_cpu
    x = np.array([[1,0],[0,2],[3,3],[5,5]],dtype=np.float32)
    q = np.zeros((1,2),dtype=np.float32)
    ids,qi = np.arange(4),np.array([-1])
    original = search(x,q,ids,qi,[4],k=1,device="cpu")[4]
    same = search(scale_features(x,np.ones(2)),scale_features(q,np.ones(2)),ids,qi,[4],k=1,device="cpu")[4]
    np.testing.assert_array_equal(original["indices"],same["indices"])
    weight = np.array([10,.1])
    result = search(scale_features(x,weight),scale_features(q,weight),ids,qi,[4],k=1,device="cpu")[4]
    assert original["indices"][0,0] == 0 and result["indices"][0,0] == 1
    assert verify_cpu(scale_features(x,weight),scale_features(q,weight),ids,qi,result,4) == 1
