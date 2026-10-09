"""GBCT target adaptation with stable physical reconstruction; no oracle scale."""
import numpy as np
from benchmarks.flame_conditioning.coordinates import state_change, inverse_state_change


def encode(initial, delta, target, interval=1e-6):
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Interval must be finite and positive")
    change = state_change(initial, delta)
    if target == "state-boxcox":
        return change
    if target != "gbct":
        raise ValueError("Unknown target")
    rate = change / interval
    return 2*np.sign(rate)*np.sqrt(np.abs(rate))


def decode(initial, coordinate, target, interval=1e-6):
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("Interval must be finite and positive")
    if target == "gbct":
        # b=.5: sign(z)*(b*abs(z))**(1/b), with a continuous zero derivative.
        coordinate = interval*.25*coordinate*np.abs(coordinate)
    elif target != "state-boxcox":
        raise ValueError("Unknown target")
    return inverse_state_change(initial, coordinate)


def differentiable_decode(initial, coordinate, target, interval=1e-6):
    from benchmarks.offline_accuracy.improve.coordinates import state_inverse
    if target == "gbct":
        coordinate = interval*.25*coordinate*coordinate.abs()
    elif target != "state-boxcox":
        raise ValueError("Unknown target")
    return state_inverse(initial, coordinate)
