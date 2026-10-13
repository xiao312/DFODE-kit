import numpy as np
import pytest
torch=pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.gradient_probe.metrics import geometry,groups,physical_counts


def test_geometry_separates_conflict_orthogonality_and_zero():
    result=geometry([torch.tensor([1.,0.]),torch.tensor([-2.,0.]),torch.tensor([0.,1.]),torch.zeros(2)])
    assert result["cosines"][0]==[1.,-1.,0.,None]


def test_counts_reconcile_group_decomposition():
    d=np.array([[0.,1e-10,1e-13]]);b=np.zeros_like(d)
    masks=groups(d,b);counts=physical_counts(d,d,b,masks)
    assert counts["all"]["passes"]==3 and counts["all"]["repaired"]==2
    assert sum(counts[k]["components"] for k in ["floor","weak_nontrivial","active_increment"])==3


def test_independent_recount_has_distinct_damage_and_repair_denominators():
    from benchmarks.offline_accuracy.paper_baseline.gradient_probe.review import recount
    d=np.array([[0.,1e-10,1e-13,2e-10]])
    old=np.array([[0.,0.,1e-13,2e-10]])
    new=np.array([[0.,1e-10,0.,2e-10]])
    result=recount(d,old,new)
    assert result["all"]["passes"]==3
    assert result["base_pass"]["components"]==3
    assert result["base_fail"]["components"]==1
    assert result["base_pass"]["damaged"]==1
    assert result["base_fail"]["repaired"]==1
