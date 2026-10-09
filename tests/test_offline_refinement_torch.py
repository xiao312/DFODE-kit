import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.offline_accuracy.refinement.coordinates import decode, differentiable_inverse
from benchmarks.offline_accuracy.refinement.fit import fit, prediction, reload_model
from benchmarks.offline_accuracy.refinement.plan import configuration
from benchmarks.offline_accuracy.refinement.verify import check_preprocessing


@pytest.mark.parametrize("target", ["budget-log", "budget-asinh"])
def test_inverse_values_and_gradients(target):
    values = np.array([-250., -1e-7, 0., 1e-7, 250.])
    z = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    actual = differentiable_inverse(z, target)
    expected, _ = decode(None, values, target, 1)
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-14)
    actual.sum().backward()
    derivative = (1e-15*np.exp(.1*abs(values)) if target == "budget-log"
                  else 1e-15*np.cosh(.1*values))
    np.testing.assert_allclose(z.grad.numpy(), derivative, rtol=2e-14)
    assert torch.isfinite(z.grad).all()
    assert z.grad[2] == 1e-15


@pytest.mark.parametrize("name", ["budget-log", "physical-budget-asinh", "residual-state-boxcox", "long-state-boxcox"])
def test_small_fit_replays_and_preserves_base(tmp_path, name):
    torch.set_num_threads(1)
    rng = np.random.default_rng(10)
    states = np.column_stack([rng.uniform(900, 1400, 32), rng.uniform(97000, 105000, 32),
                             rng.uniform(.1, .2, 32), rng.uniform(.1, .2, 32), np.zeros(32)])
    delta = np.column_stack([rng.uniform(-1e-7, 1e-7, 32), rng.uniform(-1e-9, 1e-9, 32), np.zeros(32)])
    rows = dict(states=states, delta=delta, source_indices=np.arange(32))
    config = configuration(name, 20261011)
    config.update(widths=[8, 8], updates=3, batch_size=8, validation_every=1)
    calls = []

    def base(s):
        value = np.column_stack([1e-8*s[:, 2], -1e-8*s[:, 3], np.zeros(len(s))])
        calls.append(hashlib.sha256(value.tobytes()).hexdigest())
        return value, np.zeros_like(value, dtype=bool)

    frozen = base if config["residual"] else None
    model, prep, result = fit(rows, rows, ["A", "B", "AR"], config, tmp_path, frozen)
    loaded, loaded_prep = reload_model(tmp_path, config)
    expected, _ = prediction(model, prep, states, config["target"], frozen)
    actual, _ = prediction(loaded, loaded_prep, states, config["target"], frozen)
    np.testing.assert_array_equal(actual, expected)
    assert result["updates_completed"] == 3
    assert len(result["history"]) == 3
    assert np.all(actual[:, 2] == 0)
    check_preprocessing(rows, ["A", "B", "AR"], config, loaded_prep, frozen, result)
    changed = {key: value.copy() for key, value in loaded_prep.items()}
    changed["x_scale"][0] *= 2
    with pytest.raises(AssertionError):
        check_preprocessing(rows, ["A", "B", "AR"], config, changed, frozen, result)
    if frozen:
        assert len(set(calls)) == 1
        assert result["frozen_base_training_prediction_sha256"] == calls[0]
