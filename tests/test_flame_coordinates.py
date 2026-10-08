import numpy as np
import pytest

from benchmarks.flame_conditioning.coordinates import (
    KINDS, decode, encode, input_features, inverse_state_change, standardization, state_change,
)


@pytest.mark.parametrize("kind", KINDS)
def test_target_round_trip(kind):
    initial = np.array([[0., 1e-30, .5, .2, .01]])
    delta = np.array([[1e-32, -1e-31, 1e-32, -.2, -.009]])
    scale = np.full_like(delta, 1e-8)
    encoded = encode(initial, delta, kind, scale)
    recovered, correction = decode(initial, encoded, kind, scale)
    np.testing.assert_allclose(recovered, delta, rtol=5e-13, atol=1e-45)
    assert not correction.any()


def test_stable_boxcox_retains_small_increment():
    initial = np.array([.5])
    delta = np.array([1e-32])
    assert (initial + delta)[0] == initial[0]
    assert state_change(initial, delta)[0] != 0
    recovered, _ = inverse_state_change(initial, state_change(initial, delta))
    np.testing.assert_allclose(recovered, delta, rtol=1e-14)


def test_inverse_domain_correction_is_explicit():
    delta, mask = inverse_state_change(np.array([.5, 0]), np.array([-100., -1.]))
    np.testing.assert_array_equal(mask, [True, True])
    np.testing.assert_array_equal(delta, [-.5, 0])


def test_negative_endpoint_cannot_encode():
    with pytest.raises(ValueError):
        state_change(np.array([.1]), np.array([-.2]))


def test_input_features_preserve_tiny_species_differences():
    states = np.array([[300, 101325, 1e-100, 1], [300, 101325, 2e-100, 1]])
    features = input_features(states)
    assert features[0, 2] != features[1, 2]
    offset, scale = standardization(features)
    assert np.isfinite((features - offset) / scale).all()
    assert scale[0] == 300


def test_standardization_uses_only_supplied_training_rows():
    training = np.array([[1., 2.], [3., 4.]])
    offset, scale = standardization(training)
    np.testing.assert_array_equal(offset, [2, 3])
    np.testing.assert_array_equal(scale, [1, 1])
