"""Small arithmetic, objective and local-model checks; no chemistry asset needed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from benchmarks.flame_conditioning.coordinates import inverse_state_change
from benchmarks.offline_accuracy.improve.coordinates import state_inverse, correction_inverse, physical_loss, transitions
from benchmarks.offline_accuracy.improve.plan import configuration
from benchmarks.offline_accuracy.improve import local


def test_stable_inverse_and_gradient():
    initial = np.array([[0., 1e-30, 1e-3, 1e-3, 1e-3]])
    encoded = np.array([[1e-2, 1e-8, 1e-20, -6., -.1]])
    x = torch.tensor(encoded, requires_grad=True)
    value, mask = state_inverse(torch.tensor(initial), x)
    expected, corrected = inverse_state_change(initial, encoded)
    np.testing.assert_allclose(value.detach().numpy(), expected, rtol=1e-13, atol=1e-35)
    np.testing.assert_array_equal(mask.numpy(), corrected)
    value.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_relative_chart_preserves_base_and_derivative():
    base = torch.tensor([[-1e-3, 0., 1e-32, 1e-3]], dtype=torch.float64)
    z = torch.zeros_like(base, requires_grad=True)
    result = correction_inverse(base, z)
    assert torch.equal(result, base)
    result.sum().backward()
    scale = 1e-14 + base.abs()
    torch.testing.assert_close(z.grad, torch.sqrt(scale.square()+base.square()), rtol=1e-14, atol=0)
    trial = torch.full_like(base, .3)
    torch.testing.assert_close(correction_inverse(base, trial), scale*torch.sinh(torch.asinh(base/scale)+trial))


def test_guard_penalizes_losing_base_passes():
    reference = torch.tensor([[0., 1e-4]], dtype=torch.float64)
    prediction = torch.tensor([[1e-14, 1e-4]], dtype=torch.float64, requires_grad=True)
    active = torch.tensor([True, True])
    simple = physical_loss(prediction, reference, active, reference, 0., 0.)
    guarded = physical_loss(prediction, reference, active, reference, .25, 2.)
    assert guarded > simple
    guarded.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0] > 0


def test_transition_accounting_excludes_argon():
    result = transitions(np.array([[0., 0., 0.]]), np.array([[2e-15, 1e-3, 1.]]),
                         np.array([[0., 1e-3, 0.]]), ["A", "B", "AR"])
    assert result == dict(base_pass_count=1, final_pass_count=1, pass_to_fail=1, fail_to_pass=1, components=2)


def test_config_is_bounded_and_fixed():
    config = configuration("protected-correction", 20261011)
    assert config["atol"] == 1e-15 and config["rtol"] == .1
    assert config["guard_weight"] > 0 and config["updates"] == 4000
    with pytest.raises(ValueError):
        configuration("new-unknown-recipe", 20261011)


def test_local_table_handles_constant_dimensions_and_reloads(tmp_path):
    t = np.linspace(500, 1500, 32)
    states = np.column_stack([t, np.full(32, 101325.), np.full(32, .2), np.full(32, .8)])
    delta = np.column_stack([1e-8*(t/1000), np.zeros(32)])
    p = dict(x_offset=np.array([1000., 101325., .2**.1, .8**.1]),
             x_scale=np.array([300., 5066.25, 1., 1.]), active=np.array([True, False]))
    config = configuration("local-asinh", 20261011)
    model, details = local.fit(dict(states=states, delta=delta), config, p, tmp_path)
    assert details["numerical_input_rank"] == 1
    predicted, _ = model(states)
    np.testing.assert_allclose(predicted, delta, rtol=1e-6, atol=1e-20)
    replay, _ = local.reload(tmp_path, config)(states)
    np.testing.assert_array_equal(predicted, replay)


@pytest.mark.parametrize("name", ["continue-coordinate", "finetune-tail", "protected-correction"])
def test_small_fit_preserves_frozen_base_and_replays(tmp_path, name):
    from benchmarks.flame_conditioning.train import network
    from benchmarks.flame_conditioning.coordinates import input_features, standardization, state_change
    from benchmarks.offline_accuracy.refinement.fit import weight_hash
    from benchmarks.offline_accuracy.improve import neural
    states = np.column_stack([np.linspace(800, 1200, 16), np.full(16, 101325.),
                              np.full(16, .2), np.full(16, .8)])
    delta = np.column_stack([np.linspace(1e-8, 3e-8, 16), np.zeros(16)])
    offset, scale = standardization(input_features(states))
    y_offset, y_scale = standardization(state_change(states[:, 2:], delta))
    p = dict(x_offset=offset, x_scale=scale, y_offset=y_offset, y_scale=y_scale,
             active=np.array([True, False]), target_scale=np.ones(2))
    base = network(4, 2, [8, 8], 3, torch.float32, "gelu")
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    before = weight_hash(base)
    config = configuration(name, 20261011)
    config.update(updates=2, widths=[8, 8], batch_size=8)
    data = dict(states=states, delta=delta, source_indices=np.arange(16))
    predictor, detail = neural.fit(data, data, config, base, p, tmp_path)
    assert detail["updates_completed"] == 2 and weight_hash(base) == before
    saved = neural.reload(tmp_path, config, base)
    np.testing.assert_array_equal(predictor(states)[0], saved(states)[0])


@pytest.mark.parametrize("name", ["arrhenius-heads", "arrhenius-lbfgs", "arrhenius-local"])
def test_arrhenius_inputs_fit_and_replay(tmp_path, name):
    from benchmarks.offline_accuracy.improve import arrhenius
    states = np.column_stack([np.linspace(800, 1200, 32), np.full(32, 101325.),
                              np.full(32, .2), np.full(32, .8), np.zeros(32)])
    delta = np.column_stack([np.linspace(1e-8, 3e-8, 32), np.zeros((32, 2))])
    weights = np.array([2., 28., 40.])
    config = configuration(name, 20261011)
    config.update(updates=4 if name == "arrhenius-heads" else 0, batch_size=8, widths=[4, 4])
    if name == "arrhenius-lbfgs":
        config.update(updates=4, adam_updates=2, lbfgs_steps=2)
    data = dict(states=states, delta=delta, source_indices=np.arange(32))
    values = arrhenius.features(states, weights)
    assert np.isfinite(values).all() and np.all(values[:, -1] == 0)
    active = np.array([True, False, False])
    model, detail = arrhenius.fit(data, data, dict(molecular_weights=weights), config, active, tmp_path)
    saved = arrhenius.reload(tmp_path, config)
    np.testing.assert_array_equal(model(states)[0], saved(states)[0])
    assert np.isfinite(model(states)[0]).all()


def test_state_failure_profile_is_not_component_average():
    from benchmarks.offline_accuracy.improve.diagnostics import state_profile
    profile = state_profile(np.array([[0., 0.], [0., 2e-15]]), np.zeros((2, 2)), np.array([True, True]))
    assert profile["minimum_failing_species"] == 0
    assert profile["maximum_failing_species"] == 1
    assert profile["median_failing_species"] == .5
    assert profile["failure_histogram"] == [dict(failing_species=0, states=1), dict(failing_species=1, states=1), dict(failing_species=2, states=0)]
