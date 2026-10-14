"""Stable FP64 reconstruction and training-only physical error objectives."""
import numpy as np
import torch

ATOL, RTOL, POWER = 1e-15, .1, .1


def state_inverse(initial, coordinate):
    """Differentiable counterpart of the existing stable Box-Cox inverse.

    Invalid negative endpoint powers follow the existing explicit zero-endpoint
    correction. Safe inactive branches avoid NaN gradients from torch.where.
    """
    initial, coordinate = initial.double(), coordinate.double()
    base0 = initial.pow(POWER)
    base1 = base0 + POWER * coordinate
    positive = (initial > 0) & (base1 > 0)
    safe0 = torch.where(initial > 0, base0, torch.ones_like(base0))
    ratio = POWER * coordinate / safe0
    moderate = positive & (ratio.abs() < .5)
    small_log = torch.log1p(torch.where(moderate, ratio, torch.zeros_like(ratio)))
    large_log = torch.log(torch.where(positive, base1, torch.ones_like(base1))) - torch.log(safe0)
    log_ratio = torch.where(moderate, small_log, large_log)
    changed = initial * torch.expm1(log_ratio / POWER)
    result = torch.where(positive, changed, -initial)
    result = torch.where(initial == 0, base1.clamp_min(0).pow(1/POWER), result)
    return result, base1 < 0


def correction_inverse(base, correction):
    """Base-relative asinh chart; a zero correction preserves base exactly.

    Equivalent to s*sinh(asinh(base/s)+correction), but use a stable difference
    so the initial prediction is not changed by a transform round trip.
    The scale is available at inference and contains no reference labels.
    """
    base, correction = base.double(), correction.double()
    scale = ATOL / RTOL + base.abs()
    q = base / scale
    change = scale * (2*q*torch.sinh(correction/2).square()
                      + torch.sqrt(1+q.square())*torch.sinh(correction))
    return base + change


def physical_loss(prediction, reference, active, base, tail_weight, guard_weight):
    error = (prediction[:, active]-reference[:, active]).abs()
    budget = ATOL + RTOL*reference[:, active].abs()
    ratio = error / budget
    robust = torch.log1p(ratio)
    loss = robust.mean()
    if tail_weight:
        # States fail if any species fails: retain a per-state tail term.
        loss = loss + tail_weight * robust.topk(min(8, robust.shape[1]), dim=1).values.mean()
    if guard_weight:
        passing_base = (base[:, active]-reference[:, active]).abs() <= budget
        penalty = torch.log1p(torch.relu(ratio-.5)).square()
        loss = loss + guard_weight * (penalty*passing_base).sum()/passing_base.sum().clamp_min(1)
    return loss


def transitions(base, prediction, reference, species):
    active = np.array([name != "AR" for name in species])
    truth = reference[:, active]
    budget = ATOL + RTOL*np.abs(truth)
    old = np.abs(base[:, active]-truth) <= budget
    new = np.abs(prediction[:, active]-truth) <= budget
    return dict(base_pass_count=int(old.sum()), final_pass_count=int(new.sum()),
                pass_to_fail=int((old & ~new).sum()), fail_to_pass=int((~old & new).sum()),
                components=int(old.size))
