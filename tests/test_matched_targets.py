from copy import deepcopy
from itertools import islice
from types import SimpleNamespace
import numpy as np
import pytest

from benchmarks.offline_accuracy.paper_baseline.matched_targets.plan import (
    configuration, TARGETS, OBJECTIVES, SEEDS, update_policy, batch_indices, phase_objective)
from benchmarks.offline_accuracy.paper_baseline.matched_targets.coordinates import encode, decode, preprocessing
from benchmarks.offline_accuracy.paper_baseline.matched_targets.checks import check_result, check_pair


def test_matrix_and_shared_schedule(tmp_path):
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.campaign import commands
    args = SimpleNamespace(**{key:tmp_path/key for key in ("campaign", "previous", "original", "base_root", "baseline", "output")})
    stages = commands(args)
    assert len(stages) == len({name for name, _ in stages}) == 24
    assert not args.output.exists()
    for target in TARGETS:
        for objective in OBJECTIVES:
            config = configuration(target, objective, SEEDS[0])
            assert config["updates"]*config["batch_size"] == 180000000
            for completed, rate, reset in ((0,1e-3,True),(5999,1e-3,False),(6000,1e-4,True),(12000,1e-5,True)):
                actual, actual_reset = update_policy(config, completed)
                assert actual == pytest.approx(rate) and actual_reset is reset
            assert phase_objective(config,11999) == "coordinate"
            assert phase_objective(config,12000) == objective
            a = islice(batch_indices(200000,config), 2)
            b = islice(batch_indices(200000,configuration(TARGETS[0],OBJECTIVES[0],SEEDS[0])),2)
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
    with pytest.raises(ValueError):
        configuration("unknown","coordinate",SEEDS[0])


@pytest.mark.parametrize("target", TARGETS)
def test_roundtrip_signed_tiny_and_zero(target):
    config = configuration(target,"coordinate",SEEDS[0])
    initial = np.array([[0.,1e-3,1e-3,1e-30,1e-3,1e-3]])
    delta = np.array([[1e-32,1e-25,-1e-8,1e-20,-1e-3,0.]])
    actual,_ = decode(initial,encode(initial,delta,config),config)
    np.testing.assert_allclose(actual,delta,rtol=2e-13,atol=1e-45)
    np.testing.assert_array_equal(decode(initial,encode(initial,np.zeros_like(delta),config),config)[0],np.zeros_like(delta))


def rows():
    states = np.array([[900.+i*50,100000.+i*100,.1+i*.01,.9-i*.01] for i in range(8)])
    return dict(states=states,delta=np.column_stack([np.linspace(-1e-5,2e-5,8),np.zeros(8)]),source_indices=np.arange(8))


def test_normalization_has_common_inputs_and_exact_state_control():
    from benchmarks.offline_accuracy.paper_baseline.coordinates import preprocessing as old_preprocessing
    from benchmarks.offline_accuracy.paper_baseline.matched_work.plan import extended_configuration
    training = rows()
    old = old_preprocessing(training,["H2","AR"],extended_configuration("fuel-state",SEEDS[0]))
    for target in TARGETS:
        prep = preprocessing(training,["H2","AR"],configuration(target,"coordinate",SEEDS[0]))
        for key in (old if target == "state-boxcox" else ("x_offset","x_scale","active")):
            np.testing.assert_array_equal(prep[key],old[key])
        for objective in OBJECTIVES:
            other = preprocessing(training,["H2","AR"],configuration(target,objective,SEEDS[0]))
            for key in prep:
                np.testing.assert_array_equal(prep[key],other[key])


@pytest.mark.parametrize("target", TARGETS)
def test_torch_inverse_parity_and_finite_gradient(target):
    torch = pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.coordinates import differentiable_decode
    config = configuration(target,"increment",SEEDS[0])
    initial = np.array([[0.,.1,1e-20,.1]])
    delta = np.array([[0.,1e-5,-1e-21,-.01]])
    coordinate = encode(initial,delta,config)
    value = torch.tensor(coordinate,dtype=torch.float64,requires_grad=True)
    actual = differentiable_decode(torch.tensor(initial),value,config)
    expected,_ = decode(initial,coordinate,config)
    np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=1e-12,atol=1e-40)
    actual.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_physical_objectives_differ_only_in_allowance():
    torch = pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.coordinates import physical_loss
    config = configuration("state-boxcox","increment",SEEDS[0])
    initial,reference,predicted = [torch.tensor([[v]],dtype=torch.float64) for v in (.1,1e-20,1e-6)]
    for objective,magnitude in (("increment",reference.abs()),("state",(initial+reference).abs())):
        expected = torch.log1p((predicted-reference).abs()/(1e-15+.1*magnitude)).mean()
        assert physical_loss(predicted,reference,initial,objective,config).item() == expected.item()


def result_stub():
    config = configuration("state-boxcox","coordinate",SEEDS[0])
    return dict(config=config,status="complete",training_count=200000,development_count=1023,independent_test_count=0,
        updates_completed=18000,row_presentations=180000000,effective_batch_size=10000,
        history=[dict(updates=n,row_presentations=n*10000,objective="coordinate",loss=1.,training={},development={}) for n in range(1000,18001,1000)],
        warmup_weights_sha256="warmup",initial_weights_sha256="init",preprocessing_array_sha256="prep",
        hashes={"data":"frozen"},source={"commit":"same"},baseline_result_sha256="old")


def test_changed_work_warmup_or_identity_is_rejected():
    first = result_stub()
    check_result(first)
    for key,value in (("independent_test_count",1),("row_presentations",10),("updates_completed",6000)):
        with pytest.raises(ValueError):
            check_result(dict(first,**{key:value}))
    check_pair(first,deepcopy(first),True)
    for key in ("warmup_weights_sha256","preprocessing_array_sha256","initial_weights_sha256","baseline_result_sha256"):
        with pytest.raises(ValueError):
            check_pair(first,dict(first,**{key:"changed"}),True)
    second = deepcopy(first)
    second["history"][0]["loss"] = 2.
    with pytest.raises(ValueError):
        check_pair(first,second,True)


@pytest.mark.parametrize("target", TARGETS)
def test_gpu_fit_replay_and_paired_warmup(target,tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required; no CPU training fallback")
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.fit import fit
    from benchmarks.offline_accuracy.paper_baseline.matched_targets.model import reload_model,prediction
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    training = rows()
    normalization = {key:value[:4] for key,value in training.items()}
    results = []
    for objective in OBJECTIVES:
        config = dict(configuration(target,objective,SEEDS[0]),updates=6,warmup=4,batch_size=2,
                      widths=[8,8],normalization_rows=4,diagnostics_every_updates=2)
        destination = tmp_path/objective
        destination.mkdir()
        model,prep,result = fit(training,training,normalization,["H2","AR"],config,destination)
        restored,saved = reload_model(destination,config)
        np.testing.assert_array_equal(prediction(model,prep,training["states"],config)[0],
                                      prediction(restored,saved,training["states"],config)[0])
        results.append(result)
    assert len({r["warmup_weights_sha256"] for r in results}) == 1
    assert len({r["initial_weights_sha256"] for r in results}) == 1
