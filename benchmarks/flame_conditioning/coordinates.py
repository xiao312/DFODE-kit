"""Stable target coordinates with explicit Box-Cox inverse-domain corrections."""
from __future__ import annotations

import numpy as np

from benchmarks.precision_conditioning.reference.targets import decode as decode_delta
from benchmarks.precision_conditioning.reference.targets import encode as encode_delta

KINDS = ("state-boxcox", "signed-power", "budget-linear", "scaled-asinh")
POWER = 0.1


def state_change(initial, delta):
    """Compute [Y1**lambda-Y0**lambda]/lambda without avoidable cancellation."""
    initial, delta = np.broadcast_arrays(np.asarray(initial, dtype=float), np.asarray(delta, dtype=float))
    if not np.isfinite(initial).all() or not np.isfinite(delta).all() or np.any(initial < 0):
        raise ValueError("State and change must be finite; initial state must be nonnegative")
    endpoint = initial + delta
    if np.any(endpoint < 0):
        raise ValueError("Box-Cox target requires a nonnegative endpoint")
    result = np.empty_like(initial)
    zero_initial = initial == 0
    result[zero_initial] = endpoint[zero_initial] ** POWER / POWER
    to_zero = (~zero_initial) & (endpoint == 0)
    result[to_zero] = -(initial[to_zero] ** POWER) / POWER
    active = ~(zero_initial | to_zero)
    ratio = delta[active] / initial[active]
    # Huge growth can overflow delta/Y0 even when both states are finite.
    log_ratio = np.empty_like(ratio)
    moderate = np.isfinite(ratio) & (np.abs(ratio) < .5)
    log_ratio[moderate] = np.log1p(ratio[moderate])
    log_ratio[~moderate] = np.log(endpoint[active][~moderate]) - np.log(initial[active][~moderate])
    result[active] = initial[active] ** POWER * np.expm1(POWER * log_ratio) / POWER
    return result


def inverse_state_change(initial, encoded):
    """Return increments and a mask of negative power-base corrections.

    A predicted negative Y**lambda is outside Box-Cox's range. Map that endpoint
    to zero and expose the mask; never hide this correction in model metrics.
    """
    initial, encoded = np.broadcast_arrays(np.asarray(initial, dtype=float), np.asarray(encoded, dtype=float))
    if np.any(initial < 0) or not np.isfinite(initial).all() or not np.isfinite(encoded).all():
        raise ValueError("Invalid inverse input")
    base0 = initial ** POWER
    base1 = base0 + POWER * encoded
    corrected = base1 < 0
    result = np.empty_like(initial)
    zero_initial = initial == 0
    result[zero_initial] = np.maximum(base1[zero_initial], 0) ** (1 / POWER)
    to_zero = (~zero_initial) & (base1 <= 0)
    result[to_zero] = -initial[to_zero]
    active = ~(zero_initial | to_zero)
    ratio = POWER * encoded[active] / base0[active]
    logarithm = np.empty_like(ratio)
    moderate = np.abs(ratio) < .5
    logarithm[moderate] = np.log1p(ratio[moderate])
    logarithm[~moderate] = np.log(base1[active][~moderate]) - np.log(base0[active][~moderate])
    result[active] = initial[active] * np.expm1(logarithm / POWER)
    if not np.isfinite(result).all():
        raise ValueError("Inverse produced nonfinite changes")
    return result, corrected


def encode(initial, delta, kind, asinh_scale):
    if kind == "state-boxcox":
        return state_change(initial, delta)
    scale = 1e-12 + 1e-6 * np.abs(initial) if kind == "budget-linear" else asinh_scale if kind == "scaled-asinh" else 1
    return encode_delta(delta, kind, scale=scale)


def decode(initial, encoded, kind, asinh_scale):
    if kind == "state-boxcox":
        return inverse_state_change(initial, encoded)
    scale = 1e-12 + 1e-6 * np.abs(initial) if kind == "budget-linear" else asinh_scale if kind == "scaled-asinh" else 1
    return decode_delta(encoded, kind, scale=scale), np.zeros_like(encoded, dtype=bool)


def input_features(states):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or not np.isfinite(states).all() or np.any(states[:, 2:] < 0):
        raise ValueError("Expected finite states with nonnegative mass fractions")
    # Affine-equivalent to Box-Cox after standardization, without subtracting 1
    # before representing trace species. T and pressure remain linear features.
    return np.column_stack([states[:, :2], states[:, 2:] ** POWER])


def standardization(values, center=True):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected nonempty finite training matrix")
    offset = values.mean(axis=0) if center else np.zeros(values.shape[1])
    magnitude = np.abs(values).max(axis=0)
    scale = np.sqrt(np.mean((values - offset) ** 2, axis=0))
    constant = np.ptp(values, axis=0) <= 64 * np.finfo(float).eps * magnitude
    scale[constant] = np.where(magnitude[constant] == 0, 1, magnitude[constant])
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid training standardization")
    return offset, scale
