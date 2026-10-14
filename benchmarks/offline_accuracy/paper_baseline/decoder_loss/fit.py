"""Matched GPU fitting with per-update raw gradients and a frozen numerical base."""
import json
import os
import subprocess
import time
import numpy as np
import torch
from benchmarks.offline_accuracy.paired.fit import weight_hash
from benchmarks.offline_accuracy.paired.metrics import counts
from ..matched_work.plan import batch_indices
from ..adaptive.gpu import inventory
from .model import create_models,features,prediction
from .coordinates import torch_decode,physical_loss


def occupancy_sample(step):
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    matches = [row for row in inventory() if selected in (row["index"], row["uuid"])]
    if len(matches) != 1:
        raise ValueError("Cannot identify exactly one selected GPU for occupancy telemetry")
    gpu = matches[0]
    lines = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits"],text=True,timeout=15).splitlines()
    other = [int(pid.strip()) for pid,uuid in (line.split(",") for line in lines)
             if uuid.strip()==gpu["uuid"] and int(pid.strip())!=os.getpid()]
    return dict(update=step,selected_device=selected,other_process_count=len(other),gpu=gpu)


def diagnostic(models,prep,rows,config):
    p,_ = prediction(models,prep,rows["states"],config)
    active = prep["active"]
    truth = rows["delta"][:,active]
    return counts(abs(p[:,active]-truth),1e-15+.1*abs(truth))


def fit(base,prep,base_values,training,development,normalization,config,destination):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; no CPU fallback")
    started,cpu = time.monotonic(),time.process_time()
    models = create_models(base,prep,config)
    frozen,fitted = models
    base_hash,initial_hash = weight_hash(frozen),weight_hash(fitted)
    np.testing.assert_array_equal(prediction(models,prep,training["states"],config)[0],base_values)
    x = torch.as_tensor(features(training["states"],base_values,prep,config),dtype=torch.float32,device="cuda")
    active = prep["active"]
    truth = torch.as_tensor(training["delta"][:,active],dtype=torch.float64,device="cuda")
    b = torch.as_tensor(base_values[:,active],dtype=torch.float64,device="cuda")
    optimizer = torch.optim.Adam(fitted.parameters(),lr=config["learning_rate"])
    torch.cuda.reset_peak_memory_stats()
    history,gradients,largest_gradients,occupancy = [],[],[],[]
    with (destination/"gradients.jsonl").open("x") as gradient_log:
        for completed,selected in enumerate(batch_indices(len(x),config)):
            if time.monotonic()-started>config["wall_seconds"]:
                raise TimeoutError("Decoder/loss training wall limit")
            fraction = completed/max(1,config["updates"]-1)
            rate = config["final_learning_rate"]+.5*(config["learning_rate"]-config["final_learning_rate"])*(1+np.cos(np.pi*fraction))
            optimizer.param_groups[0]["lr"] = rate
            indices = torch.as_tensor(selected,device="cuda")
            optimizer.zero_grad(set_to_none=True)
            predicted = torch_decode(b[indices],fitted(x[indices]),config)
            loss = physical_loss(predicted,truth[indices],config)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss or decoder output")
            loss.backward()
            norm = torch.sqrt(sum(p.grad.double().square().sum() for p in fitted.parameters() if p.grad is not None))
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite raw gradient norm")
            gradient = float(norm)
            largest = float(torch.stack([p.grad.abs().max() for p in fitted.parameters() if p.grad is not None]).max())
            gradients.append(gradient)
            largest_gradients.append(largest)
            gradient_log.write(json.dumps(dict(update=completed+1,raw_l2=gradient,raw_max=largest,clipped=False))+"\n")
            optimizer.step()
            step = completed+1
            if step%config["diagnostics_every_updates"]==0 or step==config["updates"]:
                record = dict(updates=step,row_presentations=step*config["batch_size"],loss=float(loss.detach()),
                    learning_rate=rate,training=diagnostic(models,prep,training,config),
                    development=diagnostic(models,prep,development,config),elapsed_wall_seconds=time.monotonic()-started)
                history.append(record)
                with (destination/"progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(record,allow_nan=False)+"\n")
                occupancy.append(occupancy_sample(step))
                print(json.dumps(record),flush=True)
    if weight_hash(frozen)!=base_hash:
        raise ValueError("Frozen base changed")
    torch.cuda.synchronize()
    torch.save(frozen.state_dict(),destination/"base-weights.pt")
    torch.save(fitted.state_dict(),destination/"weights.pt")
    torch.save(optimizer.state_dict(),destination/"optimizer.pt")
    np.savez(destination/"preprocessing.npz",**prep)
    np.save(destination/"training-indices.npy",training["source_indices"])
    np.save(destination/"normalization-indices.npy",normalization["source_indices"])
    from .diagnostics import quantiles
    return models,dict(history=history,updates_completed=step,row_presentations=step*config["batch_size"],
        initial_weights_sha256=initial_hash,frozen_base_weights_sha256=base_hash,zero_correction_identity=True,
        parameter_count=sum(p.numel() for p in fitted.parameters()),
        deployment_parameter_count=sum(p.numel() for m in models for p in m.parameters()),
        training_wall_seconds=time.monotonic()-started,training_process_seconds=time.process_time()-cpu,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),raw_gradient_norm=quantiles(np.array(gradients)),
        raw_gradient_max=quantiles(np.array(largest_gradients)),nonfinite_gradient_count=0,
        gradient_clipping_count=0,gpu_occupancy_samples=occupancy)
