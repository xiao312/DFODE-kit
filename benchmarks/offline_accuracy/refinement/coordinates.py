"""Tolerance-derived coordinates. FP64 inverses fail on overflow, never clip."""
import numpy as np

from benchmarks.flame_conditioning.coordinates import decode as old_decode, encode as old_encode
from .plan import ATOL, RTOL


def encode(initial, delta, target, scale):
    delta = np.asarray(delta, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Target must be finite")
    if target == "budget-log":
        return np.sign(delta) * np.log1p(RTOL * np.abs(delta) / ATOL) / RTOL
    if target == "budget-asinh":
        return np.arcsinh(RTOL * delta / ATOL) / RTOL
    if target == "residual":
        return delta / scale
    return old_encode(initial, delta, target, scale)


def decode(initial, coordinate, target, scale):
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if not np.isfinite(coordinate).all():
        raise ValueError("Coordinate must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        if target == "budget-log":
            value = np.sign(coordinate) * (ATOL / RTOL) * np.expm1(RTOL * np.abs(coordinate))
        elif target == "budget-asinh":
            value = (ATOL / RTOL) * np.sinh(RTOL * coordinate)
        elif target == "residual":
            value = coordinate * scale
        else:
            return old_decode(initial, coordinate, target, scale)
    if not np.isfinite(value).all():
        raise ValueError("Inverse overflow; model is invalid, not silently clipped")
    return value, np.zeros_like(value, dtype=bool)


def differentiable_inverse(coordinate, target):
    """Only new budget targets support the physical-loss ablation.

    Express signed expm1 as an odd analytic branch near zero. sign(x)*f(abs(x))
    alone gives an incorrect zero autograd derivative at x=0.
    """
    import torch
    z = coordinate.double()
    if target == "budget-asinh":
        return (ATOL / RTOL) * torch.sinh(RTOL * z)
    if target != "budget-log":
        raise ValueError("Physical loss is limited to the two budget coordinates")
    magnitude = RTOL * z.abs()
    small = magnitude < 1e-5
    safe = torch.where(small, torch.ones_like(magnitude), magnitude)
    ratio = torch.expm1(safe) / safe
    ratio = torch.where(small, 1 + magnitude / 2 + magnitude.square() / 6, ratio)
    return ATOL * z * ratio
