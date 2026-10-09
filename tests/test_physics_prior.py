import numpy as np
import pytest

from benchmarks.offline_accuracy.improve.physics_prior import frozen_step, Predictor


def test_frozen_exponential_matches_linear_production_destruction():
    initial = np.array([.1, 0., 1e-30])
    production = np.array([.2, .3, 0.])
    coefficient = np.array([2., 0., 1e8])
    h = 1e-6
    expected = np.array([0., h*.3, 1e-30*np.expm1(-100.)])
    actual = frozen_step(initial, production-coefficient*initial, coefficient*initial, h, "frozen-exponential")
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.)
    with pytest.raises(ValueError):
        frozen_step(np.zeros(1), np.ones(1), np.ones(1), h, "frozen-exponential")


def test_kinetics_controls_replay_and_zero_time():
    import cantera as ct
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1000., ct.one_atm, "H2:2,O2:1,N2:3.76"
    states = np.array([[gas.T, gas.P, *gas.Y]])
    for mode in ("rate-euler", "frozen-exponential"):
        model = Predictor("h2o2.yaml", 0., mode)
        np.testing.assert_array_equal(model(states), np.zeros((1, gas.n_species)))
        model.interval = 1e-8
        np.testing.assert_array_equal(model(states), Predictor("h2o2.yaml", 1e-8, mode)(states))
