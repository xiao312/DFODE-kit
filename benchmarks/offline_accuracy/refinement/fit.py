"""One bounded deterministic fit; physical residuals never use validation labels."""
import hashlib
import time

import numpy as np
import torch

from benchmarks.flame_conditioning.coordinates import input_features, standardization
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.metrics import tolerance_counts
from .coordinates import decode, differentiable_inverse, encode
from .plan import ATOL, COORDINATE_SCALE, RTOL


def weight_hash(model):
    return hashlib.sha256(b"".join(p.detach().double().numpy().tobytes() for p in model.parameters())).hexdigest()


def prediction(model, preprocessing, states, target, base=None):
    features = (input_features(states) - preprocessing["x_offset"]) / preprocessing["x_scale"]
    with torch.no_grad():
        # Fixed chunks bound memory and make replay independent of caller batch size.
        parts = [model(torch.as_tensor(chunk, dtype=torch.float32)).double().numpy()
                 for chunk in np.array_split(features, max(1, int(np.ceil(len(features) / 1024))))]
    coordinate = np.concatenate(parts) * preprocessing["y_scale"] + preprocessing["y_offset"]
    value, correction = decode(states[:, 2:], coordinate, target, preprocessing["target_scale"])
    if base is not None:
        original, original_correction = base(states)
        value = original.astype(np.float64) + value
        correction |= original_correction
    value[:, ~preprocessing["active"]] = 0
    correction[:, ~preprocessing["active"]] = False
    return value, correction


def fit(training, validation, species, config, destination, base=None):
    wall, cpu = time.monotonic(), time.process_time()
    states, delta = training["states"], training["delta"]
    target = config["target"]
    if (target == "residual") != (base is not None):
        raise ValueError("A residual requires exactly one frozen base")
    x_offset, x_scale = standardization(input_features(states))
    x_offset[1], x_scale[1] = 101325., 5066.25
    active = np.array([name != "AR" for name in species]) & np.any(delta != 0, axis=0)
    if not active.any():
        raise ValueError("No active output")
    labels = delta.copy()
    base_initial_hash = None
    if base is not None:
        base_prediction, _ = base(states)
        base_initial_hash = hashlib.sha256(base_prediction.tobytes()).hexdigest()
        labels -= base_prediction.astype(np.float64)
        target_scale = np.maximum(np.sqrt(np.mean(labels ** 2, axis=0)), 1e-30)
    else:
        target_scale = np.maximum(np.quantile(np.abs(delta), .9, axis=0), 1e-32)
    transformed = encode(states[:, 2:], labels, target, target_scale)
    if target.startswith("budget-") and target != "budget-linear":
        y_offset, y_scale = np.zeros(delta.shape[1]), np.full(delta.shape[1], COORDINATE_SCALE)
    elif target == "residual":
        y_offset, y_scale = np.zeros(delta.shape[1]), np.ones(delta.shape[1])
    else:
        y_offset, y_scale = standardization(transformed)
    preprocessing = dict(x_offset=x_offset, x_scale=x_scale, y_offset=y_offset,
                         y_scale=y_scale, target_scale=target_scale, active=active)
    x = torch.as_tensor((input_features(states)-x_offset)/x_scale, dtype=torch.float32)
    y = torch.as_tensor((transformed-y_offset)/y_scale, dtype=torch.float32)
    reference = torch.as_tensor(delta, dtype=torch.float64)
    mask = torch.as_tensor(active)
    model = network(x.shape[1], delta.shape[1], config["widths"], config["seed"], torch.float32, "gelu")
    initial_hash = weight_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    history = []
    probe = np.arange(min(256, len(states)))
    for step in range(1, config["updates"] + 1):
        if time.monotonic() - wall > config["wall_seconds"]:
            raise TimeoutError("Bounded fit stopped; no completed model is claimed")
        progress = (step-1)/max(config["updates"]-1, 1)
        optimizer.param_groups[0]["lr"] = config["final_learning_rate"] + .5 * (
            config["learning_rate"]-config["final_learning_rate"]) * (1+np.cos(np.pi*progress))
        indices = rng.integers(0, len(states), size=config["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        output = model(x[indices])
        coordinate_loss = (output[:, mask]-y[indices][:, mask]).abs().mean()
        physical_loss = torch.zeros((), dtype=torch.float64)
        if config["physical_weight"]:
            coordinate = output.double() * torch.as_tensor(y_scale) + torch.as_tensor(y_offset)
            physical = differentiable_inverse(coordinate, target)
            ratio = (physical-reference[indices]).abs() / (ATOL+RTOL*reference[indices].abs())
            physical_loss = torch.log1p(ratio[:, mask]).mean()
        loss = coordinate_loss + config["physical_weight"] * physical_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss; this variant failed")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradient; this variant failed")
        optimizer.step()
        if step % config["validation_every"] == 0 or step == config["updates"]:
            record = dict(step=step, batch_coordinate_loss=float(coordinate_loss.detach()),
                          batch_physical_loss=float(physical_loss.detach()))
            for split, rows, truth in (("training_probe", states[probe], delta[probe]),
                                       ("validation", validation["states"], validation["delta"])):
                predicted, _ = prediction(model, preprocessing, rows, target, base)
                record[split] = tolerance_counts(predicted[:, active], truth[:, active], ATOL, RTOL)
            history.append(record)
    if base is not None:
        after, _ = base(states)
        if hashlib.sha256(after.tobytes()).hexdigest() != base_initial_hash:
            raise ValueError("Frozen base changed during correction training")
    torch.save(model.state_dict(), destination / "weights.pt")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    np.savez(destination / "preprocessing.npz", **preprocessing)
    np.save(destination / "training-indices.npy", training["source_indices"])
    result = dict(status="complete", config=config, training_count=len(states),
                  updates_completed=step, initial_weights_sha256=initial_hash,
                  parameter_count=sum(p.numel() for p in model.parameters()), history=history,
                  training_process_seconds=time.process_time()-cpu,
                  training_wall_seconds=time.monotonic()-wall,
                  frozen_base_training_prediction_sha256=base_initial_hash)
    return model, preprocessing, result


def reload_model(directory, config):
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as arrays:
        preprocessing = {key: arrays[key] for key in arrays.files}
    model = network(len(preprocessing["x_offset"]), len(preprocessing["active"]),
                    config["widths"], config["seed"], torch.float32, "gelu")
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, preprocessing
