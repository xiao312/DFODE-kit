"""Common 18k schedule; objective arms must have exactly identical 12k warmup."""
import json
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.paired.fit import weight_hash
from ..matched_work.fit import preprocessing_hash
from ..matched_work.plan import validate_pool
from .plan import batch_indices, update_policy, phase_objective
from .coordinates import preprocessing, encode, differentiable_decode, physical_loss
from .model import diagnostic


def fit(training, development, normalization, species, config, destination):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; no CPU training fallback")
    validate_pool(len(training["states"]), config)
    if len(normalization["states"]) != config["normalization_rows"]:
        raise ValueError("Wrong normalization pool size")
    torch.cuda.synchronize()
    started, cpu = time.monotonic(), time.process_time()
    prep = preprocessing(normalization, species, config)
    active = prep["active"]
    x = torch.as_tensor((input_features(training["states"])-prep["x_offset"])/prep["x_scale"],
                        dtype=torch.float32, device="cuda")
    encoded = encode(training["states"][:, 2:], training["delta"], config)[:, active]
    y = torch.as_tensor((encoded-prep["y_offset"])/prep["y_scale"], dtype=torch.float32, device="cuda")
    initial = torch.as_tensor(training["states"][:, 2:][:, active], device="cuda", dtype=torch.float64)
    truth = torch.as_tensor(training["delta"][:, active], device="cuda", dtype=torch.float64)
    offset, scale = [torch.as_tensor(prep[k], device="cuda", dtype=torch.float64) for k in ("y_offset", "y_scale")]
    model = network(x.shape[1], y.shape[1], config["widths"], config["seed"], torch.float32, "gelu").to("cuda")
    initial_hash = weight_hash(model)
    history, schedule = [], []
    presentations, updates = 0, 0
    warmup_hash, last_rate = None, None
    torch.cuda.reset_peak_memory_stats()
    for completed, selected in enumerate(batch_indices(len(x), config)):
        if time.monotonic()-started > config["wall_seconds"]:
            raise TimeoutError("Matched-target fit reached wall limit")
        rate, reset = update_policy(config, completed)
        if reset:
            optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        optimizer.param_groups[0]["lr"] = rate
        if rate != last_rate:
            schedule.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
            last_rate = rate
        indices = torch.as_tensor(selected, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        output = model(x[indices])
        objective = phase_objective(config, completed)
        if objective == "coordinate":
            loss = (output-y[indices]).abs().mean()
        else:
            predicted = differentiable_decode(initial[indices], output.double()*scale+offset, config)
            loss = physical_loss(predicted, truth[indices], initial[indices], objective, config)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        updates = completed+1
        presentations += len(selected)
        if updates == config["warmup"]:
            warmup_hash = weight_hash(model)
        if updates % config["diagnostics_every_updates"] == 0 or updates == config["updates"]:
            record = dict(updates=updates, row_presentations=presentations, learning_rate=rate,
                          objective=objective, loss=float(loss.detach()),
                          training=diagnostic(model, prep, training, config),
                          development=diagnostic(model, prep, development, config),
                          elapsed_wall_seconds=time.monotonic()-started)
            history.append(record)
            with (destination / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            print(json.dumps({key:record[key] for key in ("updates", "objective", "loss", "elapsed_wall_seconds")}), flush=True)
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
        final_weights_sha256=weight_hash(model), warmup_weights_sha256=warmup_hash,
        preprocessing_array_sha256=preprocessing_hash(prep), schedule=schedule, history=history,
        training_wall_seconds=elapsed, training_process_seconds=time.process_time()-cpu,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
