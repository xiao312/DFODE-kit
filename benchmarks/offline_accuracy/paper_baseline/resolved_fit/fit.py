"""GPU full-batch fitting with masked direct or physical supervision."""
import json
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.train import network
from benchmarks.offline_accuracy.paired.fit import weight_hash
from ..decoder_loss.coordinates import encode, decode, torch_decode
from ..decoder_loss.diagnostics import quantiles
from .data import scores


def head(config, inputs, outputs):
    model = network(inputs, outputs, config["widths"], config["seed"], torch.float32, "gelu").to("cuda")
    with torch.no_grad():
        model[-1].weight.zero_()
        model[-1].bias.zero_()
    return model


def objective(coordinate, base, truth, mask, target, config):
    if config["fit_objective"] == "direct":
        return (coordinate.double()[mask]-target[mask]).square().mean()
    if config["fit_objective"] != "physical":
        raise ValueError("Unknown fitting objective")
    predicted = torch_decode(base, coordinate, config)
    error = (predicted-truth).abs()/(1e-15+.1*truth.abs())
    return torch.log1p(error[mask]).mean()


def fit(x, base, sample, active, species, config, destination):
    if not torch.cuda.is_available():
        raise ValueError("GPU required; CPU training is disabled")
    destination.mkdir()
    started = time.monotonic()
    cpu = time.process_time()
    model = head(config, x.shape[1], int(active.sum()))
    initial = weight_hash(model)
    tensor = lambda a,dtype:torch.as_tensor(a,dtype=dtype,device="cuda")
    features = tensor(x,torch.float32)
    b = tensor(base[:,active],torch.float64)
    truth = tensor(sample["reference"][:,active],torch.float64)
    mask = tensor(sample["resolved"][:,active],torch.bool)
    target = tensor(encode(base[:,active],sample["reference"][:,active],config),torch.float64)
    optimizer = torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    torch.cuda.reset_peak_memory_stats()

    def predict():
        with torch.no_grad():
            c = model(features).double().cpu().numpy()
        prediction = base.copy()
        prediction[:,active] = decode(base[:,active],c,config)
        return c,prediction

    np.testing.assert_array_equal(predict()[1],base)
    norms = []
    history = []
    with (destination/"gradients.jsonl").open("x") as gradients:
        for step in range(1,config["updates"]+1):
            if time.monotonic()-started > config["wall_seconds"]:
                raise TimeoutError("Small-set fit wall limit")
            fraction = (step-1)/max(1,config["updates"]-1)
            lr = config["final_learning_rate"]+.5*(config["learning_rate"]-config["final_learning_rate"])*(1+np.cos(np.pi*fraction))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(features),b,truth,mask,target,config)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training objective")
            loss.backward()
            norm = torch.sqrt(sum(p.grad.double().square().sum() for p in model.parameters()))
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite raw gradient")
            norms.append(float(norm))
            gradients.write(json.dumps(dict(update=step,raw_l2=norms[-1],clipped=False))+"\n")
            optimizer.step()
            if step % config["diagnostics_every_updates"] == 0 or step == config["updates"]:
                coordinate,predicted = predict()
                metric = scores(predicted,sample["reference"],sample["resolved"],base,species)
                record = dict(update=step,loss_before_update=float(loss.detach()),learning_rate=lr,
                    resolved=metric["resolved"],nontrivial=metric["nontrivial"],
                    coordinate_rmse=float(np.sqrt(np.mean((coordinate[mask.cpu().numpy()]-target.cpu().numpy()[mask.cpu().numpy()])**2))),
                    elapsed_seconds=time.monotonic()-started)
                history.append(record)
                with (destination/"progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(record,allow_nan=False)+"\n")
                print(json.dumps(dict(fit=destination.name,**record)),flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic()-started
    coordinate,predicted = predict()
    torch.save(model.state_dict(),destination/"weights.pt")
    torch.save(optimizer.state_dict(),destination/"optimizer.pt")
    np.savez(destination/"predictions.npz",coordinate=coordinate,prediction=predicted)
    restored = head(config,x.shape[1],int(active.sum()))
    restored.load_state_dict(torch.load(destination/"weights.pt",weights_only=True,map_location="cuda"))
    with torch.no_grad():
        replay = restored(features).double().cpu().numpy()
        gpu = torch_decode(b,tensor(replay,torch.float32),config).cpu().numpy()
    np.testing.assert_array_equal(coordinate,replay)
    np.testing.assert_allclose(gpu,predicted[:,active],rtol=2e-13,atol=1e-40)
    # Recount the GPU reconstruction separately from NumPy metric construction.
    accepted_gpu = (abs(gpu-sample["reference"][:,active]) <= 1e-15+.1*abs(sample["reference"][:,active]))
    accepted_cpu = abs(predicted-sample["reference"]) <= 1e-15+.1*abs(sample["reference"])
    np.testing.assert_array_equal(accepted_gpu[sample["resolved"][:,active]],
                                  accepted_cpu[:,active][sample["resolved"][:,active]])
    return dict(config=config,initial_weights_sha256=initial,
        parameter_count=sum(p.numel() for p in model.parameters()),
        updates_completed=step,row_presentations=step*len(x),training_wall_seconds=elapsed,
        training_process_seconds=time.process_time()-cpu,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        raw_gradient_norm=quantiles(np.asarray(norms)),gradient_clipping_count=0,
        exact_head_replay=True,cuda_numpy_pass_agreement=True,zero_correction_identity=True,
        history=history,metrics=scores(predicted,sample["reference"],sample["resolved"],base,species))
