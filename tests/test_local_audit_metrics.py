import numpy as np
from benchmarks.offline_accuracy.paper_baseline.local_function.audit_metrics import damage_slices,gate_failures


def test_damage_slices_keep_correct_denominators():
    d=np.full((4,2),1e-10);b=d.copy();b[1,0]+=1e-11;b[2,1]=0
    y=b.copy();y[0,0]=0;y[2,1]=d[2,1]
    value=damage_slices(d,b,y,np.array(["a","a","b","b"]))
    assert value["snapshot"]["a"]["damaged"]==1
    assert value["snapshot"]["b"]["repaired"]==1
    assert sum(v["base_passes"] for v in value["base_margin"].values())==7


def test_independent_gates_reject_identical_base_without_gain():
    d=np.tile([1e-13,1e-10],(128,1));b=d*.99
    policy=dict(minimum_gain_fraction=.005,minimum_group_components_for_pilot_gate=128,
                primary_group_max_drop_fraction=.005,supported_species_max_drop_fraction=.01,
                maximum_damage_rate=.02,maximum_p99_ratio=1.05)
    assert gate_failures(d,b,b,np.ones_like(d),policy,[],["a","b"])==["overall_gain"]
