"""Source-specific train-only scales and cancellation-resistant reconstruction."""
import numpy as np
from benchmarks.flame_conditioning.coordinates import input_features, state_change, inverse_state_change


def standardize(values, ddof, center=True):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) <= ddof or not np.isfinite(values).all():
        raise ValueError("Require finite training rows and enough samples")
    offset = values.mean(axis=0) if center else np.zeros(values.shape[1])
    # Standard deviation is about the sample mean even when target center is zero.
    scale = values.std(axis=0, ddof=ddof)
    constant = np.ptp(values, axis=0) == 0
    scale[constant] = 1.
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid fitted scale")
    return offset, scale, constant


def encode(initial, delta, recipe):
    if recipe == "fuel-state":
        return state_change(initial, delta)
    if recipe == "fuel-power":
        return np.sign(delta)*np.abs(delta)**.1
    raise ValueError("Unknown recipe")


def decode(initial, coordinate, recipe):
    if recipe == "fuel-state":
        return inverse_state_change(initial, coordinate)
    if recipe == "fuel-power":
        delta = np.sign(coordinate)*np.abs(coordinate)**10
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite reconstructed increment")
        return delta, np.zeros_like(delta, dtype=bool)
    raise ValueError("Unknown recipe")


def preprocessing(training, species, config):
    recipe = config["recipe"]
    states, delta = training["states"], training["delta"]
    active = np.array([name != "AR" for name in species])
    if not active.any() or np.any(states[:, 2:] < 0) or np.any(states[:, 2:]+delta < 0):
        raise ValueError("Require valid nonnegative chemistry states")
    ddof = 1 if recipe == "fuel-state" else 0
    xo, xs, xc = standardize(input_features(states), ddof)
    encoded = encode(states[:, 2:], delta, recipe)[:, active]
    yo, ys, yc = standardize(encoded, ddof, center=recipe == "fuel-state")
    return dict(x_offset=xo, x_scale=xs, x_constant=xc, y_offset=yo,
                y_scale=ys, y_constant=yc, active=active)
