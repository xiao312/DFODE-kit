"""Fixed minibatch work with a frozen training-only preprocessing pool."""
import hashlib
import json
import time

import numpy as np
import torch

from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.paired.fit import weight_hash
from ..coordinates import preprocessing, encode
from ..fit import diagnostic
from .plan import batch_indices, update_policy, validate_pool


def preprocessing_hash(prep):
    digest = hashlib.sha256()
    for name, value in sorted(prep.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fit(training, validation, normalization, species, config, destination):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; no CPU training fallback")
    validate_pool(len(training["states"]), config)
    if len(normalization["states"]) != config["normalization_rows"]:
        raise ValueError("Normalization pool has wrong size")
    torch.cuda.synchronize()
    started, cpu = time.monotonic(), time.process_time()
    prep = preprocessing(normalization, species, config)
    x = torch.as_tensor((input_features(training["states"])-prep["x_offset"])/prep["x_scale"],
                        dtype=torch.float32, device="cuda")
    encoded = encode(training["states"][:, 2:], training["delta"], config["recipe"])[:, prep["active"]]
    y = torch.as_tensor((encoded-prep["y_offset"])/prep["y_scale"], dtype=torch.float32, device="cuda")
    model = network(x.shape[1], y.shape[1], config["widths"], config["seed"], torch.float32, "gelu").to("cuda")
    initial_hash = weight_hash(model)
    history, schedule = [], []
    presentations, updates = 0, 0
    torch.cuda.reset_peak_memory_stats()
    last_rate = None
    for completed, selected in enumerate(batch_indices(len(x), config)):
        if time.monotonic()-started > config["wall_seconds"]:
            raise TimeoutError("Matched fit reached its wall limit")
        rate, reset = update_policy(config, completed)
        if reset:
            optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        optimizer.param_groups[0]["lr"] = rate
        if rate != last_rate:
            schedule.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
            last_rate = rate
        indices = torch.as_tensor(selected, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x[indices])-y[indices]).abs().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite training gradient")
        optimizer.step()
        updates = completed+1
        presentations += len(selected)
        if updates % config["diagnostics_every_updates"] == 0 or updates == config["updates"]:
            record = dict(updates=updates, row_presentations=presentations, learning_rate=rate,
                          completed_pool_passes=presentations//len(x), loss=float(loss.detach()),
                          training=diagnostic(model, prep, training, config),
                          development=diagnostic(model, prep, validation, config),
                          elapsed_wall_seconds=time.monotonic()-started)
            history.append(record)
            with (destination / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            print(json.dumps({key:record[key] for key in ("updates", "row_presentations", "loss", "elapsed_wall_seconds")}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    torch.save(model.state_dict(), destination / "weights.pt")
    torch.save(optimizer.state_dict(), destination / "optimizer.pt")
    np.savez(destination / "preprocessing.npz", **prep)
    np.save(destination / "training-indices.npy", training["source_indices"])
    np.save(destination / "normalization-indices.npy", normalization["source_indices"])
    return model.eval(), prep, dict(updates_completed=updates, row_presentations=presentations,
        completed_pool_passes=presentations//len(x), effective_batch_size=config["batch_size"],
        parameter_count=sum(p.numel() for p in model.parameters()), initial_weights_sha256=initial_hash,
        preprocessing_array_sha256=preprocessing_hash(prep), schedule=schedule, history=history,
        training_wall_seconds=elapsed, training_process_seconds=time.process_time()-cpu,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
