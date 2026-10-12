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


@pytest.mark.parametrize('objective',('physical','direct'))
def test_gpu_small_training_recipe(objective,tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    from benchmarks.offline_accuracy.paper_baseline.correction_scaling.train import fit
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import encode
    cfg = dict(configuration(20261011,10000,objective),updates=2,batch_size=4,widths=[8]*3,diagnostics_every_updates=1)
    rng = np.random.default_rng(1)
    d = np.column_stack([np.linspace(1e-12,1e-6,8),np.zeros(8)])
    b = d*2
    cache = dict(features=rng.normal(size=(8,5)).astype(np.float32),base=b,delta=d,
                 target=encode(b[:,:1],d[:,:1],cfg))
    _,r = fit(cache,cache,np.array([True,False]),cfg,tmp_path)
    assert r['updates_completed']==2 and r['row_presentations']==8 and r['raw_gradient_norm']['max']>0
