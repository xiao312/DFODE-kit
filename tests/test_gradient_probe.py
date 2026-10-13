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
