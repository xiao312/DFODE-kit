"""Paired fresh heads, fixed-work batches and training-only direct labels."""
import json
import time
import numpy as np
import torch
from benchmarks.offline_accuracy.paired.fit import weight_hash
from ..matched_work.plan import batch_indices
from ..resolved_fit.fit import head
from .objectives import objective
from ..decoder_loss.diagnostics import quantiles
from ..decoder_loss.fit import occupancy_sample
from .model import cached_prediction


def compact(model,cache,active,config):
    predicted,c = cached_prediction(model,cache,active,config)
    ref = cache["delta"][:,active]
    ratio = abs(predicted[:,active]-ref)/(1e-15+.1*abs(ref))
    nontrivial = abs(ref)>1e-15/.9
    ce = abs(c-cache["target"])
    return dict(acceptance=float((ratio<=1).mean()),nontrivial_acceptance=float((ratio[nontrivial]<=1).mean()),
        p99=float(np.quantile(ratio,.99)),coordinate_p99=float(np.quantile(ce,.99)),
        coordinate_rmse=float(np.sqrt(np.mean(ce**2))))


def fit(training,development,active,config,destination,objective_function=objective,model_factory=head):
    started = time.monotonic()
    model = model_factory(config,training["features"].shape[1],int(active.sum()))
    initial = weight_hash(model)
    np.testing.assert_array_equal(cached_prediction(model,training,active,config)[0],training["base"])
    tensor = lambda value:torch.as_tensor(value,device="cuda")
    x = tensor(training["features"])
    truth = tensor(training["delta"][:,active])
    b = tensor(training["base"][:,active])
    target = tensor(training["target"])
    mask = torch.as_tensor(training.get("eligible",np.ones_like(training["target"],dtype=bool)),device="cuda")
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    torch.cuda.reset_peak_memory_stats()
    gradients,history,occupancy = [],[],[]
    with (destination/"gradients.jsonl").open("x") as log:
        batches = (np.arange(len(x)) for _ in range(config["updates"])) if config.get("training_only_probe") else batch_indices(len(x),config)
        for completed,selected in enumerate(batches):
            if config["wall_seconds"] is not None and time.monotonic()-started > config["wall_seconds"]:
                raise TimeoutError("Fixed-work fit exceeded declared training limit")
            fraction = completed/max(1,config["updates"]-1)
            lr = config["final_learning_rate"]+.5*(config["learning_rate"]-config["final_learning_rate"])*(1+np.cos(np.pi*fraction))
            optimizer.param_groups[0]["lr"] = lr
            ix = tensor(selected)
            optimizer.zero_grad(set_to_none=True)
            loss = objective_function(model(x[ix]),b[ix],truth[ix],mask[ix],target[ix],config)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            loss.backward()
            norm = torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters()))
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite raw gradient")
            gradients.append(float(norm))
            log.write(json.dumps(dict(update=completed+1,raw_l2=gradients[-1],clipped=False))+"\n")
            optimizer.step()
            step = completed+1
            if step%config["diagnostics_every_updates"]==0:
                record = dict(update=step,loss_before_update=float(loss.detach()),learning_rate=lr,
                    training=compact(model,training,active,config),elapsed_seconds=time.monotonic()-started)
                if not config.get("training_only_probe") and development is not None:
                    record["development"] = compact(model,development,active,config)
                history.append(record)
                with (destination/"progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(record,allow_nan=False)+"\n")
                occupancy.append(occupancy_sample(step))
                print(json.dumps(record),flush=True)
    torch.cuda.synchronize()
    torch.save(model.state_dict(),destination/"weights.pt")
    torch.save(optimizer.state_dict(),destination/"optimizer.pt")
    return model,dict(initial_weights_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),
        updates_completed=step,row_presentations=step*config["batch_size"],history=history,
        training_wall_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        raw_gradient_norm=quantiles(np.array(gradients)),gradient_clipping_count=0,occupancy=occupancy)
