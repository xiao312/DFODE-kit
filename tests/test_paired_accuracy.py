import numpy as np
import pytest
from benchmarks.offline_accuracy.paired.coordinates import encode, decode
from benchmarks.offline_accuracy.paired.metrics import allowance, summarize
from benchmarks.offline_accuracy.paired.plan import configuration, POLICIES


@pytest.mark.parametrize("target", ["state-boxcox", "gbct"])
def test_roundtrip(target):
    initial = np.array([[0., 1e-3, 1e-3, 1e-30, 1e-3]])
    delta = np.array([[1e-32, 1e-25, -1e-8, 1e-20, -1e-3]])
    rebuilt, correction = decode(initial, encode(initial, delta, target), target)
    np.testing.assert_allclose(rebuilt, delta, rtol=2e-13, atol=1e-45)
    # Roundoff at the exact depletion boundary can trigger an explicit correction.
    assert not correction[0, :4].any()
    zero, _ = decode(initial, encode(initial, np.zeros_like(delta), target), target)
    np.testing.assert_array_equal(zero, np.zeros_like(delta))


def test_invalid_domain_is_visible():
    rebuilt, corrected = decode(np.array([[.1]]), np.array([[-1e6]]), "gbct")
    assert corrected.all()
    np.testing.assert_array_equal(rebuilt, [[-.1]])


def test_scaling_uses_reference_and_identical_error():
    initial = np.array([[.1, 0.]])
    truth = np.array([[1e-20, 0.]])
    predicted = np.array([[1e-6, 0.]])
    result = summarize(predicted, truth, initial, ["H2", "AR"])
    def primary(policy):
        return next(r for r in result[policy]["grid"] if r["atol"] == 1e-15 and r["rtol"] == .1)
    assert primary(POLICIES[0])["component_pass_count"] == 0
    assert primary(POLICIES[1])["component_pass_count"] == 1
    np.testing.assert_array_equal(allowance(initial, truth, POLICIES[1], 1e-15, .1), 1e-15+.1*abs(initial+truth))


def test_declared_plan():
    assert configuration("gbct-state", 20261011)["lambda_b"] == .5
    assert configuration("gbct-state", 20261011)["device"] == "cuda"
    with pytest.raises(ValueError):
        configuration("gbct-state", 3)


def test_differentiable_inverse():
    torch = pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paired.coordinates import differentiable_decode
    initial = np.array([[0., .1, 1e-20, .1]])
    coordinate = np.array([[0., 1., -1., -1e6]])
    value = torch.tensor(coordinate, dtype=torch.float64, requires_grad=True)
    predicted, corrected = differentiable_decode(torch.tensor(initial), value, "gbct")
    expected, mask = decode(initial, coordinate, "gbct")
    np.testing.assert_allclose(predicted.detach().numpy(), expected, rtol=1e-13, atol=0)
    np.testing.assert_array_equal(corrected.detach().numpy(), mask)
    predicted.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_independent_verifier_catches_changed_counts():
    pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paired.verify import check_summary
    initial = np.array([[.1, 0.], [.2, 0.]])
    truth = np.array([[1e-10, 0.], [-1e-8, 0.]])
    predicted = truth*1.01
    result = summarize(predicted, truth, initial, ["H2", "AR"])
    check_summary(predicted, truth, initial, ["H2", "AR"], result)
    result[POLICIES[1]]["grid"][0]["component_pass_count"] = 0
    with pytest.raises(AssertionError):
        check_summary(predicted, truth, initial, ["H2", "AR"], result)


def test_gpu_fit_replays(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires GPU; no CPU fallback for training")
    from benchmarks.offline_accuracy.paired.fit import fit, prediction, reload_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    states = np.array([[1000., 101325., .1, .9], [1200., 102000., .2, .8], [1300., 103000., .3, .7]])
    training = dict(states=states, delta=np.array([[1e-6, 0.], [-1e-7, 0.], [2e-6, 0.]]), source_indices=np.arange(3))
    config = dict(configuration("gbct-increment", 20261011), widths=[8, 8], updates=4,
                  warmup=2, validation_every=2, batch_size=2)
    model, prep, fitted = fit(training, ["H2", "AR"], config, tmp_path)
    expected, correction = prediction(model, prep, states, config)
    restored, restored_prep = reload_model(tmp_path, config)
    actual, actual_correction = prediction(restored, restored_prep, states, config)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_correction, correction)
    assert fitted["peak_gpu_bytes"] > 0
