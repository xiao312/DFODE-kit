"""Common train-only normalization, four targets, and FP64 inverse adapters."""
import numpy as np
from benchmarks.flame_conditioning.coordinates import input_features, state_change, inverse_state_change
from benchmarks.offline_accuracy.paired.coordinates import encode as gbct_encode, decode as gbct_decode
from ..coordinates import standardize


def encode(initial, delta, config):
    target = config["target"]
    if target == "state-boxcox":
        return state_change(initial, delta)
    if target == "signed-power":
        return np.sign(delta)*np.abs(delta)**.1
    if target == "scaled-asinh":
        return np.arcsinh(delta/config["asinh_scale"])
    if target == "gbct":
        return gbct_encode(initial, delta, target, config["interval"])
    raise ValueError("Unknown target")


def decode(initial, coordinate, config):
    target = config["target"]
    if target == "state-boxcox":
        return inverse_state_change(initial, coordinate)
    if target == "gbct":
        return gbct_decode(initial, coordinate, target, config["interval"])
    if target == "signed-power":
        delta = np.sign(coordinate)*np.abs(coordinate)**10
    elif target == "scaled-asinh":
        delta = config["asinh_scale"]*np.sinh(coordinate)
    else:
        raise ValueError("Unknown target")
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite inverse; stop rather than clip or hide it")
    return delta, np.zeros_like(delta, dtype=bool)


def differentiable_decode(initial, coordinate, config):
    from benchmarks.offline_accuracy.improve.coordinates import state_inverse
    target = config["target"]
    coordinate = coordinate.double()
    if target == "state-boxcox":
        return state_inverse(initial, coordinate)[0]
    if target == "gbct":
        return state_inverse(initial, config["interval"]*.25*coordinate*coordinate.abs())[0]
    if target == "signed-power":
        return coordinate.sign()*coordinate.abs().pow(10)
    if target == "scaled-asinh":
        return config["asinh_scale"]*coordinate.sinh()
    raise ValueError("Unknown target")


def preprocessing(training, species, config):
    states, delta = training["states"], training["delta"]
    active = np.array([name != "AR" for name in species])
    if not active.any() or np.any(states[:, 2:]+delta < 0):
        raise ValueError("Require active species and valid endpoints")
    xo, xs, xc = standardize(input_features(states), 1)
    yo, ys, yc = standardize(encode(states[:, 2:], delta, config)[:, active], 1)
    return dict(x_offset=xo, x_scale=xs, x_constant=xc, y_offset=yo,
                y_scale=ys, y_constant=yc, active=active)


def physical_loss(predicted, reference, initial, objective, config):
    import torch
    if objective not in ("increment", "state"):
        raise ValueError("Physical loss requires increment or state objective")
    magnitude = reference.abs() if objective == "increment" else (initial+reference).abs()
    budget = config["atol"]+config["rtol"]*magnitude
    return torch.log1p((predicted-reference).abs()/budget).mean()
