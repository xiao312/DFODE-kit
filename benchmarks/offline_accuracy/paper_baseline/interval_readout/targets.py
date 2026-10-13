"""Finite scale and exact accepted set are different quantities."""
import numpy as np
from benchmarks.offline_accuracy.paired.coordinates import encode


def targets(initial, delta, raw, prep):
    tau = 1e-15+.1*abs(delta)
    lower_d = np.maximum(-initial, delta-tau)
    upper_d = delta+tau
    lower = (encode(initial, lower_d, 'gbct')-prep['y_offset'])/prep['y_scale']
    upper = (encode(initial, upper_d, 'gbct')-prep['y_offset'])/prep['y_scale']
    point = (encode(initial, delta, 'gbct')-prep['y_offset'])/prep['y_scale']
    floor = 128*np.finfo(float).eps*np.maximum(1., np.maximum(abs(lower), abs(upper)))
    width = np.maximum((upper-lower)/2, floor)
    alpha = np.median(width, axis=0)
    if not np.isfinite(alpha).all() or np.any(alpha <= 0): raise ValueError('Invalid training-only scale')
    saturated = delta-tau <= -initial
    lower = np.where(saturated, -np.inf, lower)
    packed = np.stack([(lower-raw)/alpha, (upper-raw)/alpha, width/alpha, (point-raw)/alpha], axis=-1)
    return packed, alpha, dict(width_floor_components=int((width==floor).sum()), unbounded_lower_components=int(saturated.sum()))


def objective(coordinate, base, truth, eligible, target, config):
    import torch
    c = coordinate.double(); low, high, width, point = target.unbind(-1)
    if config['arm'] == 'point': error = (c-point)/width
    elif config['arm'] == 'interval': error = (torch.relu(low-c)+torch.relu(c-high))/width
    else: raise ValueError('Unknown readout objective')
    return torch.nn.functional.smooth_l1_loss(error[eligible], torch.zeros_like(error[eligible]), beta=1.)
