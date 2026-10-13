import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.local_predictability.analysis import stratified_rows, transfer


def test_selection_uses_snapshot_and_source_identity_not_target():
    rows = dict(snapshot=np.array(["b","a","b","a"]), source_indices=np.array([4,9,2,1]),
                delta=np.array([99,-5,100,0]))
    chosen = stratified_rows(rows,1)
    assert set(rows["snapshot"][chosen]) == {"a","b"}
    rows["delta"] *= -500
    np.testing.assert_array_equal(stratified_rows(rows,1), chosen)


def test_stable_neighbor_search_excludes_self_not_other_namespace():
    pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.local_predictability.neighbors import search, verify_cpu
    library = np.array([[0.],[1.],[-1.],[3.]])
    query = np.array([[0.],[0.]])
    ids, qi = np.array([0,1,2,3]), np.array([0,-1])
    result = search(library,query,ids,qi,[4],k=2,device="cpu",chunk=1)[4]
    np.testing.assert_array_equal(result["indices"], [[1,2],[0,1]])
    assert verify_cpu(library,query,ids,qi,result,4) == 2


@pytest.mark.parametrize("mode", ["target","correction"])
def test_exact_neighbor_targets_reconstruct_at_contract_precision(mode):
    truth = np.array([[0.,1e-12,-1e-5],[0.,-1e-30,1e-6]])
    base = np.array([[0.,1e-8,1e-5],[0.,1e-20,-1e-6]])
    active = np.array([False,True,True])
    pred,_ = transfer(dict(delta=truth,base=base),base,np.array([[0],[1]]),active,mode)
    assert (abs(pred-truth) <= 1e-15+.1*abs(truth)).all()


def test_constant_residual_is_reused_at_query_base():
    base = np.array([[1e-12],[2e-12]])
    truth = 1e-14*np.sinh(np.arcsinh(base/1e-14)+.2)
    query = np.array([[7e-12]])
    pred,c = transfer(dict(delta=truth,base=base),query,np.array([[0,1]]),np.array([True]),"correction")
    np.testing.assert_allclose(c,.2,atol=1e-14)
    np.testing.assert_allclose(pred,1e-14*np.sinh(np.arcsinh(query/1e-14)+.2),rtol=1e-14)
