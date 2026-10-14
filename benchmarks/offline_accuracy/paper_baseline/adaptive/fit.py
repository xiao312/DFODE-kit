"""Six thousand extra GPU updates; train-calibrated scales stay frozen."""
import json
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.offline_accuracy.paired.fit import weight_hash
from benchmarks.offline_accuracy.paired.metrics import counts, allowance
from benchmarks.offline_accuracy.paired.plan import POLICIES
from ..matched_work.plan import batch_indices
from ..matched_targets.model import prediction as base_prediction
from ..matched_targets.coordinates import differentiable_decode, physical_loss
from .plan import base_config
from .coordinates import preprocessing, scales, residual_roundtrip
from .model import create_models, features, prediction


def diagnostic(models,prep,rows,config):
    predicted,correction = prediction(models,prep,rows["states"],config)
    active = prep["active"]
    start,truth = rows["states"][:,2:][:,active],rows["delta"][:,active]
    error = abs(predicted[:,active]-truth)
    result = {policy:counts(error,allowance(start,truth,policy,1e-15,.1)) for policy in POLICIES}
    result["inverse_domain_correction_fraction"] = float(correction[:,active].mean())
    return result


def backward_mean(indices, loss_function, microbatch_size):
    """Accumulate a row-mean objective without changing group/component weights."""
    if microbatch_size<=0 or len(indices)==0:raise ValueError("Require positive batch sizes")
    total=None
    for chunk in indices.split(microbatch_size):
        loss=loss_function(chunk)*(len(chunk)/len(indices))
        if not torch.isfinite(loss):raise ValueError("Nonfinite adaptive loss")
        loss.backward()
        total=loss.detach() if total is None else total+loss.detach()
    return total


def fit(base,base_prep,training,development,normalization,config,destination,*,fixed_preprocessing=None):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; no CPU fallback")
    torch.cuda.synchronize()
    started,cpu = time.monotonic(),time.process_time()
    bc = base_config(config["seed"])
    base_values,_ = base_prediction(base,base_prep,training["states"],bc)
    if fixed_preprocessing is None:
        norm_values,_ = base_prediction(base,base_prep,normalization["states"],bc)
        prep = preprocessing(base_prep,norm_values,normalization["delta"],config)
    else:
        prep={key:value.copy() for key,value in fixed_preprocessing.items()}
        for key,value in base_prep.items():np.testing.assert_array_equal(prep["base_"+key],value)
    active = prep["active"]
    roundtrip = residual_roundtrip(base_values[:,active],training["delta"][:,active])
    if roundtrip["max_error_over_allowance"] > .01:
        raise ValueError("Residual arithmetic consumes more than 1% of tolerance")
    models = create_models(base,prep,config)
    frozen,fitted = models
    base_hash,initial_hash = weight_hash(frozen),weight_hash(fitted)
    np.testing.assert_array_equal(prediction(models,prep,training["states"],config)[0],base_values)
    x_values = ((input_features(training["states"])-base_prep["x_offset"])/base_prep["x_scale"]
                if config["arm"] == "continue" else features(training["states"],base_values,prep,config))
    x = torch.as_tensor(x_values,dtype=torch.float32,device="cuda")
    truth = torch.as_tensor(training["delta"][:,active],dtype=torch.float64,device="cuda")
    start = torch.as_tensor(training["states"][:,2:][:,active],dtype=torch.float64,device="cuda")
    b = torch.as_tensor(base_values[:,active],dtype=torch.float64,device="cuda")
    scale = torch.as_tensor(scales(base_values[:,active],prep,config),dtype=torch.float64,device="cuda")
    offset,ys = [torch.as_tensor(base_prep[k],dtype=torch.float64,device="cuda") for k in ("y_offset","y_scale")]
    optimizer = torch.optim.Adam(fitted.parameters(),lr=config["learning_rate"],foreach=config.get("optimizer_foreach"))
    torch.cuda.reset_peak_memory_stats()
    history = []
    for completed,selected in enumerate(batch_indices(len(x),config)):
        if time.monotonic()-started > config["wall_seconds"]:
            raise TimeoutError("Adaptive fit wall limit reached")
        fraction = completed/max(1,config["updates"]-1)
        rate = config["final_learning_rate"]+.5*(config["learning_rate"]-config["final_learning_rate"])*(1+np.cos(np.pi*fraction))
        optimizer.param_groups[0]["lr"] = rate
        indices = torch.as_tensor(selected,device="cuda")
        optimizer.zero_grad(set_to_none=True)
        def objective(selected):
            output = fitted(x[selected])
            predicted = (differentiable_decode(start[selected],output.double()*ys+offset,bc)
                         if config["arm"] == "continue" else b[selected]+scale[selected]*output.double())
            return physical_loss(predicted,truth[selected],start[selected],"increment",config)
        loss=backward_mean(indices,objective,config.get("microbatch_size",len(indices)))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in fitted.parameters()):
            raise ValueError("Nonfinite adaptive gradient")
        optimizer.step()
        step = completed+1
        if step % config["diagnostics_every_updates"] == 0 or step == config["updates"]:
            record = dict(updates=step,row_presentations=step*config["batch_size"],loss=float(loss.detach()),
                learning_rate=rate,training=diagnostic(models,prep,training,config),
                elapsed_wall_seconds=time.monotonic()-started)
            if development is not None:record["development"]=diagnostic(models,prep,development,config)
            history.append(record)
            with (destination/"progress.jsonl").open("a") as handle:
                handle.write(json.dumps(record,allow_nan=False)+"\n")
            print(json.dumps({k:record[k] for k in ("updates","loss","elapsed_wall_seconds")}),flush=True)
    if weight_hash(frozen) != base_hash:
        raise ValueError("Frozen base changed")
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    torch.save(frozen.state_dict(),destination/"base-weights.pt")
    torch.save(fitted.state_dict(),destination/"weights.pt")
    torch.save(optimizer.state_dict(),destination/"optimizer.pt")
    np.savez(destination/"preprocessing.npz",**prep)
    np.save(destination/"training-indices.npy",training["source_indices"])
    if fixed_preprocessing is None:np.save(destination/"normalization-indices.npy",normalization["source_indices"])
    fitted.eval()
    fitted_count = sum(p.numel() for p in fitted.parameters())
    base_count = sum(p.numel() for p in frozen.parameters())
    return models,prep,dict(history=history,updates_completed=step,row_presentations=step*config["batch_size"],
        initial_weights_sha256=initial_hash,frozen_base_weights_sha256=base_hash,zero_correction_identity=True,
        residual_arithmetic=roundtrip,parameter_count=fitted_count,
        deployment_parameter_count=fitted_count+(base_count if config["arm"] != "continue" else 0),
        training_wall_seconds=elapsed,training_process_seconds=time.process_time()-cpu,
        peak_gpu_bytes=torch.cuda.max_memory_allocated())
