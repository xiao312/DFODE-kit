"""Epoch-based GPU fitting, with source-specific schedule and bounded diagnostics."""
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.paired.fit import weight_hash
from benchmarks.offline_accuracy.paired.metrics import counts, allowance
from benchmarks.offline_accuracy.paired.plan import POLICIES
from .coordinates import preprocessing, encode, decode
from .plan import epoch_policy


def prediction(model, prep, states, config):
    features = (input_features(states)-prep["x_offset"])/prep["x_scale"]
    with torch.no_grad():
        chunks = [model(torch.as_tensor(x, dtype=torch.float32, device="cuda")).double().cpu().numpy()
                  for x in np.array_split(features, max(1, int(np.ceil(len(states)/1024))))]
    coordinate = np.concatenate(chunks)*prep["y_scale"]+prep["y_offset"]
    active = prep["active"]
    delta, correction = decode(states[:, 2:][:, active], coordinate, config["recipe"])
    output, mask = np.zeros_like(states[:, 2:]), np.zeros_like(states[:, 2:], dtype=bool)
    output[:, active], mask[:, active] = delta, correction
    return output, mask


def reload_model(directory, config):
    with np.load(directory / "preprocessing.npz", allow_pickle=False) as arrays:
        prep = {key: arrays[key] for key in arrays.files}
    model = network(len(prep["x_offset"]), int(prep["active"].sum()), config["widths"],
                    config["seed"], torch.float32, "gelu")
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    return model.to("cuda").eval(), prep


def diagnostic(model, prep, rows, config):
    predicted, corrections = prediction(model, prep, rows["states"], config)
    active = prep["active"]
    truth, start = rows["delta"][:, active], rows["states"][:, 2:][:, active]
    error = abs(predicted[:, active]-truth)
    result = {policy: counts(error, allowance(start, truth, policy, 1e-15, .1)) for policy in POLICIES}
    result["inverse_domain_correction_fraction"] = float(corrections[:, active].mean())
    return result


def fit(training, validation, species, config, destination):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; CPU training fallback is disabled")
    torch.cuda.synchronize()
    started, cpu = time.monotonic(), time.process_time()
    prep = preprocessing(training, species, config)
    x = torch.as_tensor((input_features(training["states"])-prep["x_offset"])/prep["x_scale"],
                        dtype=torch.float32, device="cuda")
    encoded = encode(training["states"][:, 2:], training["delta"], config["recipe"])[:, prep["active"]]
    y = torch.as_tensor((encoded-prep["y_offset"])/prep["y_scale"], dtype=torch.float32, device="cuda")
    model = network(x.shape[1], y.shape[1], config["widths"], config["seed"], torch.float32, "gelu").to("cuda")
    initial_hash = weight_hash(model)
    rng = np.random.default_rng(config["seed"])
    batch = min(config["batch_size"], len(x))
    updates, presentations = 0, 0
    history = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(config["epochs"]):
        rate, reset = epoch_policy(config, epoch)
        if reset:
            optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        optimizer.param_groups[0]["lr"] = rate
        permutation = rng.permutation(len(x))
        limit = len(x)//batch*batch if config["recipe"] == "fuel-state" else len(x)
        total_loss, seen = 0., 0
        for begin in range(0, limit, batch):
            if time.monotonic()-started > config["wall_seconds"]:
                raise TimeoutError("Fit time limit reached; do not treat partial results as complete")
            indices = torch.as_tensor(permutation[begin:begin+batch], device="cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[indices])-y[indices]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite coordinate loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            updates += 1
            presentations += len(indices)
            total_loss += float(loss.detach())*len(indices)
            seen += len(indices)
        if (epoch+1) % config["diagnostics_every"] == 0 or epoch+1 == config["epochs"]:
            record = dict(epoch=epoch+1, updates=updates, row_presentations=presentations,
                          lr=rate, training_epoch_loss=total_loss/seen,
                          training=diagnostic(model, prep, training, config),
                          development=diagnostic(model, prep, validation, config),
                          elapsed_wall_seconds=time.monotonic()-started)
            history.append(record)
            import json
            with (destination / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
            print(json.dumps({key: record[key] for key in ("epoch", "updates", "lr", "training_epoch_loss", "elapsed_wall_seconds")}), flush=True)
    torch.cuda.synchronize()
    training_wall = time.monotonic()-started
    torch.save(model.state_dict(), destination / "weights.pt")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    np.savez(destination / "preprocessing.npz", **prep)
    np.save(destination / "training-indices.npy", training["source_indices"])
    return model.eval(), prep, dict(epochs_completed=config["epochs"], updates_completed=updates,
        row_presentations=presentations, requested_batch_size=config["batch_size"], effective_batch_size=batch,
        parameter_count=sum(p.numel() for p in model.parameters()), initial_weights_sha256=initial_hash,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(), training_wall_seconds=training_wall,
        training_process_seconds=time.process_time()-cpu, history=history)
