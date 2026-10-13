"""One frozen forward with a label-free affine output correction."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from ..adaptive.model import base_preprocessing
from ..adaptive.plan import base_config
from ..matched_targets.coordinates import decode
from ..correction_scaling.model import fixed_batches, coordinates


def normalized_hidden(hidden, scales):
    value = hidden.double()
    offset = torch.as_tensor(scales['hidden_offset'], device=value.device)
    scale = torch.as_tensor(scales['hidden_scale'], device=value.device)
    return ((value-offset)/scale).float()


def collect(base, prep, states):
    p = base_preprocessing(prep); hs = []; zs = []
    with torch.no_grad():
        for count, chunk in fixed_batches(states):
            x = (input_features(chunk)-p['x_offset'])/p['x_scale']
            h = base[:-1](torch.as_tensor(x, dtype=torch.float32, device='cuda'))
            zs.append(base[-1](h).double().cpu().numpy()[:count]); hs.append(h.cpu().numpy()[:count])
    return np.concatenate(hs), np.concatenate(zs)


def reconstruct(states, raw, prep, seed):
    p = base_preprocessing(prep); active = prep['active']
    d, mask = decode(states[:,2:][:,active], raw*p['y_scale']+p['y_offset'], base_config(seed))
    value = np.zeros_like(states[:,2:]); value[:,active] = d
    return value, mask


def cached_prediction(head, cache, active, config):
    correction = coordinates(head, cache['features'])
    raw = cache['raw']+cache['output_scale']*correction
    return reconstruct(cache['states'], raw, cache['preprocessing'], config['seed'])[0], correction


def predict(base, head, prep, scales, states, seed):
    p = base_preprocessing(prep); raw_parts = []; c_parts = []; base_parts = []
    with torch.no_grad():
        for count, chunk in fixed_batches(states):
            x = (input_features(chunk)-p['x_offset'])/p['x_scale']
            h = base[:-1](torch.as_tensor(x, dtype=torch.float32, device='cuda'))
            raw = base[-1](h).double().cpu().numpy()
            c = head(normalized_hidden(h,scales)).double().cpu().numpy()
            base_parts.append(raw[:count]); raw_parts.append((raw+scales['output_scale']*c)[:count]); c_parts.append(c[:count])
    value, mask = reconstruct(states, np.concatenate(raw_parts), prep, seed)
    anchor, anchor_mask = reconstruct(states, np.concatenate(base_parts), prep, seed)
    return value, anchor, np.concatenate(c_parts), mask, anchor_mask
