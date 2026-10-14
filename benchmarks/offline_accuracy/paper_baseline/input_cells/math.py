"""Input-cell construction and conservative GBCT output preimages."""
import numpy as np
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.offline_accuracy.paired.coordinates import encode


def neural_input(state, prep):
    return ((input_features(np.asarray(state)[None, :])[0]-prep['x_offset'])/prep['x_scale']).astype(np.float32)


def collision(state, prep, buffer_index, excluded):
    """Largest mass exchange inside one-quarter of available rounding-cell room."""
    state = np.asarray(state, dtype=float)
    x = neural_input(state, prep)
    lower = (np.nextafter(x, np.float32(-np.inf)).astype(float)+x.astype(float))/2
    upper = (np.nextafter(x, np.float32(np.inf)).astype(float)+x.astype(float))/2
    yl = np.maximum(0, lower[2:]*prep['x_scale'][2:]+prep['x_offset'][2:])**10
    yu = np.maximum(0, upper[2:]*prep['x_scale'][2:]+prep['x_offset'][2:])**10
    y = state[2:]; choices = []
    for j in range(len(y)):
        if j in excluded or j == buffer_index or y[j] <= 0: continue
        for direction in (-1, 1):
            room = min(yu[j]-y[j], y[buffer_index]-yl[buffer_index]) if direction == 1 else min(y[j]-yl[j], yu[buffer_index]-y[buffer_index])
            choices.append((max(0., room), j, direction))
    room, species, direction = max(choices, default=(0., -1, 1))
    for halvings in range(9):
        change = direction*.25*room/(2**halvings)
        candidate = state.copy()
        candidate[2+species] += change; candidate[2+buffer_index] -= change
        if (np.array_equal(neural_input(candidate, prep).view('uint32'), x.view('uint32'))
            and np.all(candidate[2:] >= 0) and not np.array_equal(candidate, state)
            and abs(candidate[2:].sum()-y.sum()) <= 4*np.finfo(float).eps):
            return candidate, dict(species_index=species, buffer_index=buffer_index,
                                   change=change, halvings=halvings, room=room)
    return None, dict(reason='No nonzero exact collision under fixed construction')


def output_interval(initial, reference, uncertainty, prep):
    """Union over empirical reference interval, including variable allowance.

    Zero endpoint saturation makes the lower preimage unbounded. Outward
    128-epsilon margins protect strict separation from ordinary FP64 roundoff;
    they are not a formal transcendental-function error bound.
    """
    initial, reference, uncertainty = np.broadcast_arrays(initial, reference, uncertainty)
    if not np.isfinite([initial, reference, uncertainty]).all() or np.any(initial < 0) or np.any(uncertainty < 0):
        raise ValueError('Invalid reference interval')
    reflo, refhi = reference-uncertainty, reference+uncertainty
    low = reflo-1e-15-.1*abs(reflo)
    high = refhi+1e-15+.1*abs(refhi)
    valid = high >= -initial
    low_value = np.maximum(low, -initial)
    high_value = np.maximum(high, -initial)
    lower = (encode(initial, low_value, 'gbct')-prep['y_offset'])/prep['y_scale']
    upper = (encode(initial, high_value, 'gbct')-prep['y_offset'])/prep['y_scale']
    margin = 128*np.finfo(float).eps*np.maximum(1., np.maximum(abs(lower), abs(upper)))
    lower -= margin; upper += margin
    lower = np.where(low <= -initial, -np.inf, lower)
    return lower, upper, valid
