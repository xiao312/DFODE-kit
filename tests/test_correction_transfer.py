import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.correction_transfer.selection import unseen_indices,audit_selection
from benchmarks.offline_accuracy.paper_baseline.correction_transfer.metrics import score,preservation
from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import decode


def test_unseen_selection_is_disjoint_fixed_and_snapshot_balanced():
    snapshots = np.tile(np.array(['a','b','c','d']),20)
    selected = unseen_indices(snapshots,start=8,per_snapshot=5)
    np.testing.assert_array_equal(selected,unseen_indices(snapshots,start=8,per_snapshot=5))
    assert len(selected)==len(np.unique(selected))==20 and min(selected)>=8
    assert np.unique(snapshots[selected],return_counts=True)[1].tolist()==[5]*4
    with pytest.raises(ValueError,match='support'):
        unseen_indices(snapshots,start=70,per_snapshot=5)


def test_damage_and_repair_denominators():
    truth = np.array([1.,1.,1.,1.])
    base = np.array([1.,1.,2.,2.])
    predicted = np.array([1.,2.,1.,2.])
    s = score(predicted,truth,base)
    assert s['repair_rate']==s['damage_rate']==.5
    assert s['pass_to_fail']==s['fail_to_pass']==1 and s['net_pass_change']==0


def test_intervals_and_zero_damping():
    cfg = dict(decoder='asinh',scale_floor=1e-14)
    rng = np.random.default_rng(7)
    d = rng.choice([-1.,1.],200)*10.**rng.uniform(-30,-3,200)
    d = np.r_[d,0.]
    b = d+(1e-15+.1*abs(d))*rng.uniform(-1,1,len(d))
    np.testing.assert_array_equal(decode(b,np.zeros_like(b),cfg),b)
    c = rng.normal(0,.15,len(d))
    result = preservation(b,d,c,cfg)
    assert result['interval_disagreements']==0
    assert sum(r['base_successes'] for r in result['bins'])==len(d)
    assert sum(r['damaged'] for r in result['bins'])==int((abs(decode(b,c,cfg)-d)>1e-15+.1*abs(d)).sum())


def test_anchor_controls_match_without_reusing_tail_states():
    data = dict(states=np.array([[300+i,1e5,.5,.5] for i in range(24)]),
        delta=np.tile([1e-8,0.],(24,1)),base=np.tile([1e-8,0.],(24,1)),
        source_indices=np.arange(24),snapshot=np.repeat(['a','b'],12))
    data['base'][[0,1,12,13],0]=-1e-8
    tails = {11:[dict(source_row=i,training_row=i,species='H2',snapshot=str(data['snapshot'][i]),
        reference_bin='(1e-10,1e-8]',temperature_K=float(data['states'][i,0]),pressure_Pa=1e5)
        for i in (0,1,12,13)]}
    records = audit_selection(tails,{11:data},['H2','AR'],per_snapshot=2,control_stride=2)
    assert records==audit_selection(tails,{11:data},['H2','AR'],per_snapshot=2,control_stride=2)
    assert len({r['source_row'] for r in records})==6
    controls = [r for r in records if r['role']=='control']
    assert len(controls)==2 and not {r['source_row'] for r in controls}&{0,1,12,13}
    assert all(r['temperature_gap_K']>0 for r in controls)


def test_reference_score_flags_discrepancies_and_uncertainty():
    pytest.importorskip('cantera')
    from benchmarks.offline_accuracy.paper_baseline.correction_transfer.audit_worker import reference_scores
    state = dict(Y=[.1,.9])
    checks = dict(tight=dict(delta=[1e-8,0.]),tighter=dict(delta=[1e-8,0.]),
                  **{'direct-tighter':dict(delta=[1e-8,0.])})
    result = reference_scores(state,np.array([1e-8,0.]),np.array([-1e-8,0.]),checks,np.array([True,False]),0)
    assert result['anchor']['qualified'] and result['anchor']['base_sign_reversal']
    assert result['anchor']['nominal_sign_match'] and result['anchor']['sign_stable']
    checks['tight']['delta'][0]=-1e-8
    result = reference_scores(state,np.array([-1e-8,0.]),np.array([-1e-8,0.]),checks,np.array([True,False]),0)
    assert not result['anchor']['qualified'] and not result['anchor']['sign_stable']
    assert result['anchor']['nominal_difference_in_allowances']>1
