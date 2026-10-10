"""Target-aware inference and fixed-interval diagnostics, with no reference at inference."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.offline_accuracy.paired.metrics import counts, allowance
from benchmarks.offline_accuracy.paired.plan import POLICIES
from ..fit import reload_model
from .coordinates import decode


def prediction(model, prep, states, config):
    features = (input_features(states)-prep["x_offset"])/prep["x_scale"]
    with torch.no_grad():
        chunks = [model(torch.as_tensor(x, dtype=torch.float32, device="cuda")).double().cpu().numpy()
                  for x in np.array_split(features, max(1, int(np.ceil(len(states)/1024))))]
    coordinate = np.concatenate(chunks)*prep["y_scale"]+prep["y_offset"]
    active = prep["active"]
    delta, correction = decode(states[:, 2:][:, active], coordinate, config)
    output, mask = np.zeros_like(states[:, 2:]), np.zeros_like(states[:, 2:], dtype=bool)
    output[:, active], mask[:, active] = delta, correction
    return output, mask


def diagnostic(model, prep, rows, config):
    predicted, corrections = prediction(model, prep, rows["states"], config)
    active = prep["active"]
    truth, start = rows["delta"][:, active], rows["states"][:, 2:][:, active]
    error = abs(predicted[:, active]-truth)
    result = {policy: counts(error, allowance(start, truth, policy, 1e-15, .1)) for policy in POLICIES}
    result["inverse_domain_correction_fraction"] = float(corrections[:, active].mean())
    return result
