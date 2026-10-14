"""Warm-start physical objectives and zero-initialized relative corrections."""
import copy
import time

import numpy as np
import torch

from benchmarks.flame_conditioning.coordinates import input_features, state_change
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.metrics import tolerance_counts
from benchmarks.offline_accuracy.refinement.fit import prediction as base_prediction
from .coordinates import correction_inverse, physical_loss, state_inverse


class Predictor:
    def __init__(self, model, preprocessing, config, frozen_model):
        self.model, self.preprocessing = model, preprocessing
        self.config, self.frozen_model = config, frozen_model

    def __call__(self, states):
        p = self.preprocessing
        features = (input_features(states)-p["x_offset"])/p["x_scale"]
        values, masks = [], []
        # Fixed chunking is shared with the saved base's inference.
        for indices in np.array_split(np.arange(len(states)), max(1, int(np.ceil(len(states)/1024)))):
            with torch.no_grad():
                output = self.model(torch.as_tensor(features[indices], dtype=torch.float32))
                if self.config["correction"]:
                    base, corrected = base_prediction(self.frozen_model, p, states[indices], "state-boxcox")
                    value = correction_inverse(torch.as_tensor(base), output).numpy()
                else:
                    coordinate = output.double()*torch.as_tensor(p["y_scale"]) + torch.as_tensor(p["y_offset"])
                    value, corrected = state_inverse(torch.as_tensor(states[indices, 2:]), coordinate)
                    value, corrected = value.numpy(), corrected.numpy()
                value[:, ~p["active"]], corrected[:, ~p["active"]] = 0., False
                values.append(value)
                masks.append(corrected)
        value = np.concatenate(values)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite prediction; no clipping fallback")
        return value, np.concatenate(masks)


def fit(training, validation, config, frozen_model, preprocessing, destination):
    wall, cpu = time.monotonic(), time.process_time()
    states, truth = training["states"], training["delta"]
    p = preprocessing
    features = (input_features(states)-p["x_offset"])/p["x_scale"]
    base, _ = base_prediction(frozen_model, p, states, "state-boxcox")
    if config["correction"]:
        model = network(features.shape[1], truth.shape[1], config["widths"], config["seed"], torch.float32, "gelu")
        with torch.no_grad():
            model[-1].weight.zero_()
            model[-1].bias.zero_()
    else:
        model = copy.deepcopy(frozen_model)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    predict = Predictor(model, p, config, frozen_model)
    initial_prediction, _ = predict(states)
    if config["correction"]:
        np.testing.assert_array_equal(initial_prediction, base)
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(truth)
    y0 = torch.as_tensor(states[:, 2:])
    base_tensor = torch.as_tensor(base)
    active = torch.as_tensor(p["active"])
    target = torch.as_tensor((state_change(states[:, 2:], truth)-p["y_offset"])/p["y_scale"], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"])
    history = []
    for step in range(1, config["updates"]+1):
        if time.monotonic()-wall > config["wall_seconds"]:
            raise TimeoutError("Fit exceeded declared wall limit")
        progress = (step-1)/max(1, config["updates"]-1)
        optimizer.param_groups[0]["lr"] = config["final_learning_rate"] + .5*(
            config["learning_rate"]-config["final_learning_rate"])*(1+np.cos(np.pi*progress))
        indices = rng.integers(0, len(states), config["batch_size"])
        output = model(x[indices])
        if config["name"] == "continue-coordinate":
            loss = (output[:, active]-target[indices][:, active]).abs().mean()
        else:
            if config["correction"]:
                physical = correction_inverse(base_tensor[indices], output)
            else:
                coordinate = output.double()*torch.as_tensor(p["y_scale"]) + torch.as_tensor(p["y_offset"])
                physical, _ = state_inverse(y0[indices], coordinate)
            loss = physical_loss(physical, y[indices], active, base_tensor[indices],
                                 config["tail_weight"], config["guard_weight"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % config["validation_every"] == 0 or step == config["updates"]:
            record = dict(step=step, batch_loss=float(loss.detach()))
            for name, rows in (("training_probe", {key: value[:256] for key, value in training.items()}),
                               ("validation", validation)):
                predicted, _ = predict(rows["states"])
                record[name] = tolerance_counts(predicted[:, p["active"]], rows["delta"][:, p["active"]], 1e-15, .1)
            history.append(record)
    model.eval()
    torch.save(model.state_dict(), destination / "weights.pt")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    np.savez(destination / "preprocessing.npz", **p)
    return predict, dict(history=history, updates_completed=step,
                         parameter_count=sum(v.numel() for v in model.parameters()),
                         fit_process_seconds=time.process_time()-cpu, fit_wall_seconds=time.monotonic()-wall)


def reload(directory, config, frozen_model):
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as arrays:
        p = {key: arrays[key] for key in arrays.files}
    model = network(len(p["x_offset"]), len(p["active"]), config["widths"], config["seed"], torch.float32, "gelu")
    model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True, map_location="cpu"))
    model.eval()
    return Predictor(model, p, config, frozen_model)
