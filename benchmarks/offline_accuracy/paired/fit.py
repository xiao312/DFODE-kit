"""Matched two-phase fit; train-only scales and fixed final checkpoints."""
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features, standardization
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.refinement.fit import weight_hash
from .coordinates import encode, decode, differentiable_decode


def preprocessing(training, species, config):
    states, delta = training["states"], training["delta"]
    xo, xs = standardization(input_features(states))
    xo[1], xs[1] = 101325., 5066.25
    yo, ys = standardization(encode(states[:, 2:], delta, config["target"], config["interval"]))
    active = np.array([s != "AR" for s in species]) & np.any(delta != 0, axis=0)
    return dict(x_offset=xo, x_scale=xs, y_offset=yo, y_scale=ys, active=active)


def prediction(model, prep, states, config):
    x = (input_features(states)-prep["x_offset"])/prep["x_scale"]
    with torch.no_grad():
        parts = [model(torch.as_tensor(chunk, dtype=torch.float32)).double().numpy()
                 for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/1024))))]
    coordinate = np.concatenate(parts)*prep["y_scale"]+prep["y_offset"]
    predicted, correction = decode(states[:, 2:], coordinate, config["target"], config["interval"])
    predicted[:, ~prep["active"]] = 0
    correction[:, ~prep["active"]] = False
    return predicted, correction


def reload_model(directory, config):
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as arrays:
        prep = {k: arrays[k] for k in arrays.files}
    model = network(len(prep["x_offset"]), len(prep["active"]), config["widths"],
                    config["seed"], torch.float32, "gelu")
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, prep


def fit(training, species, config, destination):
    wall, cpu = time.monotonic(), time.process_time()
    states, delta = training["states"], training["delta"]
    prep = preprocessing(training, species, config)
    x = torch.as_tensor((input_features(states)-prep["x_offset"])/prep["x_scale"], dtype=torch.float32)
    encoded = encode(states[:, 2:], delta, config["target"], config["interval"])
    y = torch.as_tensor((encoded-prep["y_offset"])/prep["y_scale"], dtype=torch.float32)
    truth = torch.as_tensor(delta, dtype=torch.float64)
    initial = torch.as_tensor(states[:, 2:], dtype=torch.float64)
    offset, scale = torch.as_tensor(prep["y_offset"]), torch.as_tensor(prep["y_scale"])
    mask = torch.as_tensor(prep["active"])
    model = network(x.shape[1], y.shape[1], config["widths"], config["seed"], torch.float32, "gelu")
    initial_hash = weight_hash(model)
    rng = np.random.default_rng(config["seed"])
    history = []
    warmup_hash = None
    for step in range(1, config["updates"]+1):
        if time.monotonic()-wall > config["wall_seconds"]:
            raise TimeoutError("Fit wall limit exceeded; retain failed run")
        second = step > config["warmup"]
        if step in (1, config["warmup"]+1):
            optimizer = torch.optim.Adam(model.parameters())
        phase_step = step-config["warmup"] if second else step
        phase_length = config["updates"]-config["warmup"] if second else config["warmup"]
        prefix = "phase_two_" if second else ""
        low, high = config[prefix+"final_learning_rate"], config[prefix+"learning_rate"]
        optimizer.param_groups[0]["lr"] = low+.5*(high-low)*(1+np.cos(np.pi*(phase_step-1)/(phase_length-1)))
        indices = rng.integers(0, len(states), config["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        output = model(x[indices])
        coordinate_loss = (output[:, mask]-y[indices][:, mask]).abs().mean()
        loss = coordinate_loss
        physical_loss = None
        if second and config["objective"] != "coordinate":
            predicted, _ = differentiable_decode(initial[indices], output.double()*scale+offset,
                                                   config["target"], config["interval"])
            magnitude = truth[indices].abs() if config["objective"] == "increment" else (initial[indices]+truth[indices]).abs()
            budget = config["atol"]+config["rtol"]*magnitude
            physical_loss = torch.log1p((predicted-truth[indices]).abs()[:, mask]/budget[:, mask]).mean()
            loss = physical_loss
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        if step == config["warmup"]:
            warmup_hash = weight_hash(model)
        if step % config["validation_every"] == 0:
            record = dict(step=step, coordinate_loss=float(coordinate_loss.detach()),
                          physical_loss=None if physical_loss is None else float(physical_loss.detach()))
            history.append(record)
            print(record, flush=True)
    torch.save(model.state_dict(), destination / "weights.pt")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    np.savez(destination / "preprocessing.npz", **prep)
    np.save(destination / "training-indices.npy", training["source_indices"])
    result = dict(updates_completed=step, initial_weights_sha256=initial_hash, warmup_weights_sha256=warmup_hash,
        history=history, parameter_count=sum(p.numel() for p in model.parameters()),
        training_wall_seconds=time.monotonic()-wall, training_process_seconds=time.process_time()-cpu)
    return model, prep, result
