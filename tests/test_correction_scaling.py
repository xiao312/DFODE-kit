import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.correction_scaling.coverage import magnitude_bins,profile,support,label_check
from benchmarks.offline_accuracy.paper_baseline.correction_scaling.plan import configuration
from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import decode


def test_bins_are_disjoint_and_include_exact_edges():
    values = np.array([0.,1e-32,1e-15,np.nextafter(1e-15,np.inf),1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1.])
    np.testing.assert_array_equal(magnitude_bins(values),[0,1,1,2,2,3,4,5,6,7,8])
    np.testing.assert_array_equal(magnitude_bins(-values),magnitude_bins(values))
    with pytest.raises(ValueError):
        magnitude_bins(np.array([np.nan]))


def test_component_grain_sources_and_sparse_support():
    rows = dict(states=np.array([[900.,1e5,.1,.9],[1000.,1e5,.2,.8],[1100.,1e5,.3,.7]]),
        delta=np.array([[1e-12,0.],[-1e-12,0.],[1e-6,0.]]),source_indices=np.arange(3),snapshot=np.array(['a','a','b']))
    p = profile(rows,['H2','AR'])
    g = next(g for g in p['groups'] if g['bin']=='(1e-14,1e-12]')
    assert g['components']==g['distinct_states']==2 and g['distinct_sources']==1
    assert g['positive']==g['negative']==1
    assert sum(g['components'] for g in p['groups'])==3
    assert len(support(p,p))==2
    rows['source_indices'][1]=0
    with pytest.raises(ValueError,match='Duplicate'):
        profile(rows,['H2','AR'])


def test_label_screen_never_certifies_unaudited_or_unresolved():
    d = np.array([[1e-12,0.]])
    active = np.array([True,False])
    assert label_check(d,d,np.zeros_like(d),active)['passed']
    assert not label_check(d+1e-12,d,np.zeros_like(d),active)['passed']
    q = label_check(d,d,np.ones_like(d),active)
    assert not q['passed'] and q['checked_components']==0


def test_finite_coordinate_bound_for_both_signs_and_all_scales():
    cfg = configuration(20261011,10000,'direct')
    d = np.r_[0.,np.logspace(-300,-1,500),-np.logspace(-300,-1,500)].reshape(-1,1)
    for error in (-np.log1p(.1),-.05,0.,.05,np.log1p(.1)):
        p = decode(d,np.full_like(d,error),cfg)
        assert np.max(abs(p-d)/(1e-15+.1*abs(d)))<=1+1e-12


def test_decoder_agreement_near_zero_and_pass_boundary():
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.verify import compare_decoders
    a = np.array([6e-22])
    assert compare_decoders(a,a+1.6e-33,a)['identical_pass_decisions']
    with pytest.raises(ValueError):
        compare_decoders(a,a+1e-20,a)
    with pytest.raises(AssertionError):
        compare_decoders(np.array([1e-15]),np.array([1e-15+1e-26]),np.array([0.]))


def test_loss_concentration_and_progression_rule():
    pytest.importorskip('torch')
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.review import fitting_pressure,progression
    c = np.ones((10,10)); c[0,0]=100
    mask = np.ones_like(c,dtype=bool); mask[0,0]=False
    q = fitting_pressure(c,mask)
    assert q['top_component_count']==1 and q['top_squared_loss_share']==pytest.approx(10000/10099)
    physical = dict(development=dict(acceptance=.7,nontrivial_acceptance=.69))
    direct = dict(development=dict(acceptance=.71,base_acceptance=.7,nontrivial_acceptance=.70,base_nontrivial_acceptance=.69))
    assert progression(physical,direct)
    direct['development']['nontrivial_acceptance']=.68
    assert not progression(physical,direct)


def test_gpu_fixed_batch_subset_identity():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    from benchmarks.flame_conditioning.train import network
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.model import coordinates
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = network(9,3,[16,16],20261011,torch.float32,'gelu').to('cuda')
    x = np.random.default_rng(3).normal(size=(2100,9)).astype(np.float32)
    selected = np.array([2099,4,1025,1,199])
    np.testing.assert_array_equal(coordinates(model,x)[selected],coordinates(model,x[selected]))


@pytest.mark.parametrize('objective,probe',(('physical',False),('direct',False),('direct',True),('coordinate-sl1',False)))
def test_gpu_small_training_recipe(objective,probe,tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.train import fit
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import encode
    cfg = dict(configuration(20261011,10000,objective),updates=2,batch_size=8 if probe else 4,
               widths=[8]*3,diagnostics_every_updates=1,training_only_probe=probe)
    rng = np.random.default_rng(1)
    d = np.column_stack([np.linspace(1e-12,1e-6,8),np.zeros(8)])
    b = d*2
    cache = dict(features=rng.normal(size=(8,5)).astype(np.float32),base=b,delta=d,
                 target=encode(b[:,:1],d[:,:1],cfg))
    _,r = fit(cache,cache,np.array([True,False]),cfg,tmp_path)
    assert r['updates_completed']==2 and r['row_presentations']==(16 if probe else 8) and r['raw_gradient_norm']['max']>0
    assert ('development' in r['history'][0]) is not probe


def test_coordinate_sl1_formula_gradient_and_historical_config():
    torch = pytest.importorskip('torch')
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.objectives import objective
    cfg = configuration(20261011,10000,'coordinate-sl1')
    old = configuration(20261011,10000,'direct')
    assert {k:v for k,v in cfg.items() if k not in ('coordinate_beta','fit_objective')} == {
        k:v for k,v in old.items() if k!='fit_objective'}
    beta = np.log1p(.1)
    e = torch.tensor([-30.,-beta,-beta/2,0.,beta/2,beta,30.],dtype=torch.float64,requires_grad=True)
    mask = torch.ones_like(e,dtype=torch.bool)
    loss = objective(e,None,None,mask,torch.zeros_like(e),cfg)
    expected = np.where(abs(e.detach().numpy())<beta,e.detach().numpy()**2/(2*beta),abs(e.detach().numpy())-beta/2).mean()
    assert float(loss.detach())==pytest.approx(expected)
    loss.backward()
    np.testing.assert_allclose(e.grad.numpy()*len(e),[-1,-1,-.5,0,.5,1,1])
    with pytest.raises(ValueError,match='restricted'):
        configuration(20261011,50000,'coordinate-sl1')


def test_tail_partition_and_constant_mean():
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.tail_metrics import relationships,selected_tail,constant_scores
    b = np.array([[1e-8,-1e-12,0.,1e-12,1e-12]])
    d = np.array([[1e-12,1e-12,1e-12,1e-8,1e-12]])
    assert relationships(b,d).tolist()==[['same-sign overprediction','sign reversal','one or both zero',
                                         'same-sign underprediction','same-sign equal magnitude']]
    target = np.array([[1.,4.],[3.,6.]])
    rows,cols = selected_tail(target)
    assert rows.tolist()==[1] and cols.tolist()==[1]
    cfg = configuration(20261011,10000,'direct')
    result = constant_scores(np.zeros_like(target),np.zeros_like(target),target,target.mean(axis=0),cfg)
    assert result['coordinate_mse']==1 and result['zero_coordinate_mse']==15.5


def test_larger_rung_requires_passing_hash_bound_review(tmp_path):
    pytest.importorskip('torch')
    import json
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.gate import require_previous,validate_record
    from benchmarks.flame_conditioning.extract import sha256
    assert require_previous(10000,None,None,None,None) is None
    with pytest.raises(ValueError,match='requires'):
        require_previous(50000,None,None,None,None)
    path = tmp_path/'review.json'
    path.write_text(json.dumps(dict(status='verified',advance=False,size=10000)))
    with pytest.raises(ValueError,match='hash'):
        require_previous(50000,tmp_path,tmp_path,path,'wrong')
    with pytest.raises(ValueError,match='passing'):
        require_previous(50000,tmp_path,tmp_path,path,sha256(path))
    with pytest.raises(ValueError,match='both'):
        validate_record(dict(status='verified',advance=True,size=10000,fits=[]),50000)


def test_larger_gate_recounts_and_rejects_changed_artifact(tmp_path,monkeypatch):
    pytest.importorskip('torch')
    import json
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling import gate,review,run
    from benchmarks.flame_conditioning.extract import sha256
    record = dict(status='verified',advance=True,size=10000,fits=[])
    results = {}
    for seed in (20261011,20261012):
        for objective in ('physical','direct'):
            path = tmp_path/'10000'/f'{seed}-{objective}'
            path.mkdir(parents=True)
            (path/'result.json').write_text('{}')
            summary = dict(seed=seed,objective=objective,result_sha256=sha256(path/'result.json'),
                development=dict(acceptance=.72 if objective=='direct' else .7,base_acceptance=.7,
                                 nontrivial_acceptance=.71,base_nontrivial_acceptance=.7))
            record['fits'].append(summary)
            results[path.name] = (summary,dict(config=configuration(seed,10000,objective),
                initial_weights_sha256='weights',prepared_manifest_sha256='prepared',
                training_count=10000,row_presentations=60000000,source='source'))
    path = tmp_path/'review.json'
    path.write_text(json.dumps(record))
    monkeypatch.setattr(run,'prepared_manifest',lambda *args:dict(species=['H2','AR']))
    monkeypatch.setattr(review,'recount',lambda directory,*args:results[directory.name])
    assert gate.require_previous(50000,tmp_path,tmp_path,path,sha256(path))['size']==10000
    (tmp_path/'10000'/'20261012-direct'/'result.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='result hash'):
        gate.require_previous(50000,tmp_path,tmp_path,path,sha256(path))
