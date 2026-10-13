"""State-only base and correction inference, with one shared frozen forward pass."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from ..adaptive.model import base_preprocessing, features
from ..adaptive.plan import base_config
from ..matched_targets.coordinates import decode as base_decode
from ..correction_scaling.model import fixed_batches, coordinates, base_prediction
from ..decoder_loss.coordinates import decode
from ..correction_scaling.plan import configuration

ARMS = {"A": (119, [256, 256, 256], 177210),
        "B": (919, [256, 256, 256], 382010),
        "C": (119, [800, 256, 256], 381754)}


def config(seed, arm):
    result = configuration(seed, 10000, "coordinate-sl1")
    result.update(experiment="hidden-feature-access-v1", arm=arm, widths=ARMS[arm][1],
                  wall_seconds=None)
    return result


def base_and_hidden(base, prep, states, seed):
    p = base_preprocessing(prep)
    predictions, hidden = [], []
    for count, chunk in fixed_batches(states):
        x = (input_features(chunk) - p["x_offset"]) / p["x_scale"]
        with torch.no_grad():
            h = base[:-1](torch.as_tensor(x, dtype=torch.float32, device="cuda"))
            z = base[-1](h).double().cpu().numpy() * p["y_scale"] + p["y_offset"]
        d, _ = base_decode(chunk[:, 2:][:, prep["active"]], z, base_config(seed))
        value = np.zeros_like(chunk[:, 2:])
        value[:, prep["active"]] = d
        predictions.append(value[:count])
        hidden.append(h.cpu().numpy()[:count])
    return np.concatenate(predictions), np.concatenate(hidden)


def make_features(states, b, hidden, prep, cfg, scales):
    standard = features(states, b, prep, cfg).astype(np.float32)
    if cfg["arm"] != "B":
        return standard
    extra = ((hidden.astype(np.float64) - scales["hidden_offset"]) / scales["hidden_scale"]).astype(np.float32)
    return np.column_stack([standard, extra])


def predict(base, head, prep, scales, states, cfg):
    if cfg["arm"] == "B":
        b, hidden = base_and_hidden(base, prep, states, cfg["seed"])
    else:
        b, _ = base_prediction(base, base_preprocessing(prep), states, cfg["seed"])
        hidden = None
    x = make_features(states, b, hidden, prep, cfg, scales)
    c = coordinates(head, x)
    output = b.copy()
    output[:, prep["active"]] = decode(b[:, prep["active"]], c, cfg)
    return output, c
