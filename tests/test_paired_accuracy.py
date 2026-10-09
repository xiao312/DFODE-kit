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
