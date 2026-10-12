from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.decoder_loss.plan import configuration,ARMS,SEEDS
from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import encode,decode
from benchmarks.offline_accuracy.paper_baseline.decoder_loss.diagnostics import oracle,tails


def test_matrix_and_same_budget(tmp_path):
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.campaign import commands
    args = SimpleNamespace(**{k:tmp_path/k for k in ("campaign","previous","original","base_root","baseline","output")})
    stages = commands(args)
    assert [name for name,_ in stages]==[f"{s}-{a}" for s in SEEDS for a in ARMS]
    assert not args.output.exists()
    for arm in ARMS:
        cfg = configuration(arm,SEEDS[0])
        assert cfg["updates"]*cfg["batch_size"]==60000000
        assert cfg["coordinate_divisor"]==1 and cfg["gradient_clipping"] is None


@pytest.mark.parametrize("decoder",("linear","asinh"))
def test_zero_identity_roundtrip_and_branch_continuity(decoder):
    cfg = configuration(decoder+"-log",SEEDS[0])
    b = np.array([[0.,1e-32,-1e-32,1e-8,-1e-8,1e-3]])
    d = np.array([[1e-32,-1e-32,1e-20,1e-12,1e-12,1e-12]])
    np.testing.assert_array_equal(decode(b,np.zeros_like(b),cfg),b)
    restored = decode(b,encode(b,d,cfg),cfg)
    assert np.max(abs(restored-d)/(1e-15+.1*abs(d)))<1e-5
    for c in (-.5,.5):
        low,high = [decode(b,np.full_like(b,c+sign*1e-12),cfg) for sign in (-1,1)]
        np.testing.assert_allclose(low,high,rtol=1e-10,atol=1e-23)


def test_motivating_example_is_conditioning_not_float32_impossibility():
    b,d = np.array([[1e-8]]),np.array([[1e-12]])
    for decoder,tolerance in (("linear",1.01e-5),("asinh",.10099495)):
        cfg = configuration(decoder+"-log",SEEDS[0])
        result = oracle(b,d,["H2"],cfg)
        assert result["fp32"]["pass_count"]==1
        assert result["local_allowed_coordinate_error"]["p50"]==pytest.approx(tolerance,rel=1e-5)
        assert result["fp32_neighbor_best"]["pass_count"]==1


def test_extreme_cancellation_oracle_and_empty_event_denominators():
    b,d = np.array([[1e-3]]),np.array([[1e-20]])
    linear = oracle(b,d,["H2"],configuration("linear-log",SEEDS[0]))
    asinh = oracle(b,d,["H2"],configuration("asinh-log",SEEDS[0]))
    assert linear["fp32"]["pass_count"]==0
    assert asinh["fp32"]["pass_count"]==1
    detail = tails(np.array([[1e-8,-2e-3]]),np.array([[1e-12,1e-3]]),["H2","O2"])
    selected = next(r for r in detail["overprediction"] if r["epsilon"]==1e-12 and r["eta"]==1e-8)
    assert selected["events"]==selected["components"]==1
    assert detail["sign"]["wrong_sign_count"]==1
    assert detail["overprediction"][0]["frequency"] is None
    assert "resolved_nontrivial" not in detail
    qualified = tails(d,d,["H2"],np.full_like(d,np.inf))
    assert qualified["resolved_nontrivial"]["components"]==0
    assert qualified["resolved_nontrivial"]["acceptance"] is None


def test_torch_decoder_zero_gradient_and_loss_formulas():
    torch = pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.coordinates import torch_decode,physical_loss
    b = torch.tensor([[0.,1e-32,1e-8,-1e-8]],dtype=torch.float64)
    for decoder in ("linear","asinh"):
        cfg = configuration(decoder+"-log",SEEDS[0])
        c = torch.zeros_like(b,requires_grad=True)
        p = torch_decode(b,c,cfg)
        np.testing.assert_array_equal(p.detach().numpy(),b.numpy())
        p.sum().backward()
        expected = 1e-14+abs(b.numpy()) if decoder=="linear" else np.hypot(b.numpy(),1e-14)
        np.testing.assert_allclose(c.grad.numpy(),expected,rtol=1e-15)
        for magnitude in (1e-8,.5,.500001,-9.21,10.):
            c = torch.full_like(b,magnitude,requires_grad=True)
            p = torch_decode(b,c,cfg)
            np.testing.assert_allclose(p.detach().numpy(),decode(b.numpy(),c.detach().numpy(),cfg),rtol=1e-13,atol=1e-40)
            p.sum().backward()
            assert torch.isfinite(c.grad).all()
    truth = torch.tensor([[0.,1e-12,1e-6]],dtype=torch.float64)
    u = torch.tensor([[.2,-1.,4.]],dtype=torch.float64)
    p = truth+u*(1e-15+.1*abs(truth))
    for loss in ("log","huber"):
        cfg = configuration("linear-"+loss,SEEDS[0])
        expected = torch.log1p(abs(u)).mean() if loss=="log" else torch.tensor((.5*.2**2+.5+3.5)/3)
        assert physical_loss(p,truth,cfg).item()==pytest.approx(expected.item(),rel=1e-7)


@pytest.mark.parametrize("arm",ARMS)
def test_gpu_fit_and_checkpoint_replay(arm,tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required; no CPU training fallback")
    from benchmarks.flame_conditioning.train import network
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.coordinates import preprocessing as base_prep
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.model import prediction as base_prediction
    from benchmarks.offline_accuracy.paper_baseline.adaptive.coordinates import preprocessing
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.plan import base_config
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.fit import fit
    from benchmarks.offline_accuracy.paper_baseline.decoder_loss.model import prediction,reload_model
    from benchmarks.offline_accuracy.paired.fit import weight_hash
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    states = np.array([[900.+i*50,1e5+i*100,.1+i*.01,.9-i*.01] for i in range(8)])
    rows = dict(states=states,delta=np.column_stack([np.linspace(-1e-5,2e-5,8),np.zeros(8)]),source_indices=np.arange(8))
    norm = {k:v[:4] for k,v in rows.items()}
    cfg = dict(configuration(arm,SEEDS[0]),updates=3,batch_size=2,widths=[8,8],diagnostics_every_updates=1)
    bp = base_prep(norm,["H2","AR"],base_config(SEEDS[0]))
    base = network(4,1,[800]*4,SEEDS[0],torch.float32,"gelu").to("cuda")
    digest = weight_hash(base)
    bv,_ = base_prediction(base,bp,states,base_config(SEEDS[0]))
    nv,_ = base_prediction(base,bp,norm["states"],base_config(SEEDS[0]))
    prep = preprocessing(bp,nv,norm["delta"],cfg)
    models,result = fit(base,prep,bv,rows,rows,norm,cfg,tmp_path)
    restored,saved = reload_model(tmp_path,cfg)
    np.testing.assert_array_equal(prediction(models,prep,states,cfg)[0],prediction(restored,saved,states,cfg)[0])
    assert digest==weight_hash(models[0])==weight_hash(base)
    assert result["zero_correction_identity"] and result["gradient_clipping_count"]==0
    assert result["raw_gradient_norm"]["max"]>0
