from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.adaptive.plan import configuration,SEEDS,ARMS,BASE_HASHES
from benchmarks.offline_accuracy.paper_baseline.adaptive.coordinates import calibrate,scales,residual_roundtrip,preprocessing
from benchmarks.offline_accuracy.paper_baseline.adaptive.campaign import commands,check_result


def test_finite_queue_and_pinned_work(tmp_path):
    args = SimpleNamespace(**{key:tmp_path/key for key in ("campaign","previous","original","base_root","baseline","output")})
    stages = commands(args)
    assert [name for name,_ in stages] == [f"{seed}-{arm}" for seed in SEEDS for arm in ARMS]
    assert not args.output.exists()
    for arm in ARMS:
        cfg = configuration(arm,SEEDS[0])
        assert cfg["updates"]*cfg["batch_size"] == 60000000
    with pytest.raises(ValueError):
        configuration("unknown",SEEDS[0])


def test_calibration_bounds_zero_and_sparse_fallback():
    cfg = configuration("calibrated",SEEDS[0])
    base = np.zeros((256,3))
    truth = np.tile([0.,1e-14,1e-8],(256,1))
    prep = calibrate(base,truth,cfg)
    np.testing.assert_array_equal(prep["alpha"],np.tile([[.01],[1.],[100.]],(1,8)))
    assert np.all(prep["calibration_counts"][:,0] == 256)
    assert not prep["calibration_counts"][:,1:].any()
    assert np.isfinite(scales(base,prep,cfg)).all()
    assert np.all(scales(base,prep,cfg)>0)
    with pytest.raises(ValueError):
        calibrate(base,truth[:1],cfg)


def test_conditional_scale_interpolation_is_continuous_and_not_label_dependent():
    cfg = dict(configuration("calibrated",SEEDS[0]),minimum_bin_count=2)
    base = np.array([0.,0.,1e-10,1e-10])[:,None]
    truth = base+np.array([.1,.1,10.,10.])[:,None]*(1e-14+abs(base))
    prep = calibrate(base,truth,cfg)
    assert prep["alpha"][0,0] == pytest.approx(.1)
    assert prep["alpha"][0,2] == pytest.approx(10.)
    query = 1e-14*np.expm1(np.log(10.)*np.array([3-1e-8,3.,3+1e-8]))[:,None]
    factor = scales(query,prep,cfg)/(1e-14+abs(query))
    np.testing.assert_allclose(factor,np.broadcast_to(factor[1],factor.shape),rtol=1e-7)
    local = scales(query,prep,dict(cfg,arm="local"))
    np.testing.assert_array_equal(local,1e-14+abs(query))
    before = scales(query,prep,cfg).copy()
    truth[:] = 1e10
    np.testing.assert_array_equal(before,scales(query,prep,cfg))


def test_residual_roundtrip_and_unchanged_contract():
    base = np.array([[1e-32,0.,-1e-12,.1]])
    truth = np.array([[2e-32,1e-40,1e-14,.1000000000001]])
    result = residual_roundtrip(base,truth)
    assert result["max_error_over_allowance"] < 1e-8
    cfg = configuration("local",SEEDS[0])
    result = dict(config=cfg,status="complete",training_count=200000,development_count=1023,
        independent_test_count=0,updates_completed=6000,row_presentations=60000000,zero_correction_identity=True,
        baseline_result_sha256=BASE_HASHES[SEEDS[0]],history=[dict(updates=i) for i in range(1000,6001,1000)],
        residual_arithmetic=dict(max_error_over_allowance=0.))
    check_result(result)
    for key,value in (("updates_completed",1),("independent_test_count",1),("baseline_result_sha256","changed")):
        with pytest.raises(ValueError):
            check_result(dict(result,**{key:value}))
    altered = deepcopy(result)
    altered["config"]["atol"] = 1e-12
    with pytest.raises(ValueError):
        check_result(altered)


def test_idle_gpu_rejects_other_users_even_at_low_utilization():
    from benchmarks.offline_accuracy.paper_baseline.adaptive.gpu import idle,inventory
    from unittest.mock import patch
    outputs = ["GPU-busy\n", "0, GPU-busy, 10, 0\n1, GPU-free, 20, 1\n"]
    with patch("subprocess.check_output",side_effect=outputs):
        rows = inventory()
    assert not idle(rows[0])
    assert idle(rows[1])
    assert not idle(dict(rows[1],memory_MiB=256))
    assert not idle(dict(rows[1],utilization=10))


@pytest.mark.parametrize("arm",ARMS)
def test_gpu_fit_frozen_base_zero_identity_and_replay(arm,tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required; no CPU training fallback")
    from benchmarks.flame_conditioning.train import network
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.coordinates import preprocessing as base_preprocessing
    from benchmarks.offline_accuracy.paper_baseline.adaptive.plan import base_config
    from benchmarks.offline_accuracy.paper_baseline.adaptive.fit import fit
    from benchmarks.offline_accuracy.paper_baseline.adaptive.model import prediction,reload_model
    from benchmarks.offline_accuracy.paired.fit import weight_hash
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    states = np.array([[900.+i*50,100000.+i*100,.1+i*.01,.9-i*.01] for i in range(8)])
    rows = dict(states=states,delta=np.column_stack([np.linspace(-1e-5,2e-5,8),np.zeros(8)]),source_indices=np.arange(8))
    norm = {k:v[:4] for k,v in rows.items()}
    cfg = dict(configuration(arm,SEEDS[0]),updates=3,batch_size=2,widths=[8,8],diagnostics_every_updates=1)
    bp = base_preprocessing(norm,["H2","AR"],base_config(SEEDS[0]))
    base = network(4,1,[800]*4,SEEDS[0],torch.float32,"gelu").to("cuda")
    original = weight_hash(base)
    models,prep,result = fit(base,bp,rows,rows,norm,cfg,tmp_path)
    restored,saved = reload_model(tmp_path,cfg)
    np.testing.assert_array_equal(prediction(models,prep,states,cfg)[0],prediction(restored,saved,states,cfg)[0])
    assert original == weight_hash(base) == weight_hash(models[0])
    assert result["zero_correction_identity"] is True
    assert result["updates_completed"] == 3
    assert np.isfinite([row["loss"] for row in result["history"]]).all()
    for key in prep:
        np.testing.assert_array_equal(prep[key],saved[key])
