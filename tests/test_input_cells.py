import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.input_cells.math import collision, neural_input, output_interval
from benchmarks.offline_accuracy.paired.coordinates import decode


def test_exact_collision():
    state = np.array([1200., 101325., .2, .3, .5])
    prep = dict(x_offset=np.zeros(5), x_scale=np.ones(5))
    other, info = collision(state, prep, 2, [])
    assert other is not None and info['change'] != 0
    np.testing.assert_array_equal(neural_input(state, prep).view('uint32'), neural_input(other, prep).view('uint32'))
    assert abs(other[2:].sum()-1) < 1e-15


def test_saturated_preimage():
    prep = dict(y_offset=np.zeros(2), y_scale=np.ones(2))
    low, high, valid = output_interval(np.array([0., 1e-16]), np.zeros(2), np.zeros(2), prep)
    assert np.isneginf(low).all() and valid.all()
    y, _ = decode(np.array([0., 1e-16]), np.array([-1e8, -1e8]), 'gbct')
    assert (abs(y) <= 1e-15).all()


@pytest.mark.parametrize('reference', [-1e-7, 0., 1e-7])
def test_uncertainty_union(reference):
    y = np.array([1e-3]); d = np.array([reference]); u = np.array([1e-10])
    prep = dict(y_offset=np.array([2.]), y_scale=np.array([3.]))
    lo, hi, ok = output_interval(y, d, u, prep)
    decoded_lo, _ = decode(y, lo*3+2, 'gbct')
    decoded_hi, _ = decode(y, hi*3+2, 'gbct')
    assert ok.all()
    for r in np.linspace(reference-u[0], reference+u[0], 21):
        assert decoded_lo[0] <= r-1e-15-.1*abs(r)
        assert decoded_hi[0] >= r+1e-15+.1*abs(r)


def test_invalid_uncertainty():
    with pytest.raises(ValueError):
        output_interval([.1], [0.], [-1.], dict(y_offset=[0.], y_scale=[1.]))
