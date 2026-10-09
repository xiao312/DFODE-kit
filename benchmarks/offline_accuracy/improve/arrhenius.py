"""Physically motivated inputs and small independent species heads.

Inspired by latent-asinh surface-kinetics models; adapted to finite-time gas
chemistry increments. This is neither the same task nor a paper reproduction.
"""
import time

import numpy as np
from scipy.interpolate import RBFInterpolator
import torch

from benchmarks.flame_conditioning.coordinates import standardization
from benchmarks.offline_accuracy.metrics import tolerance_counts
from .coordinates import physical_loss


def features(states, molecular_weights, floor=1e-30):
    states = np.asarray(states, dtype=float)
    if not np.isfinite(states).all() or np.any(states[:, :2] <= 0) or np.any(states[:, 2:] < 0):
        raise ValueError("Require positive T/P and nonnegative fractions")
    amount = states[:, 2:]/molecular_weights
    mole_fraction = amount / amount.sum(axis=1, keepdims=True)
    reduced_pressure = states[:, 1:2]/101325.
    return np.column_stack([1000/states[:, 0], np.log(reduced_pressure[:, 0]),
                            np.log1p(mole_fraction*reduced_pressure/floor)])


class SpeciesHeads(torch.nn.Module):
    """Batched independent MLPs: no hidden features shared across species."""
    def __init__(self, inputs, outputs, widths, seed):
        super().__init__()
        torch.manual_seed(seed)
        dims = [inputs, *widths, 1]
        self.weights, self.biases = torch.nn.ParameterList(), torch.nn.ParameterList()
        for before, after in zip(dims[:-1], dims[1:]):
            bound = 1/np.sqrt(before)
            self.weights.append(torch.nn.Parameter(torch.empty(outputs, before, after).uniform_(-bound, bound)))
            self.biases.append(torch.nn.Parameter(torch.empty(outputs, 1, after).uniform_(-bound, bound)))

    def forward(self, x):
        value = x.unsqueeze(0).expand(self.weights[0].shape[0], -1, -1)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = torch.bmm(value, weight) + bias
            if index < len(self.weights)-1:
                value = torch.tanh(value)
        return value.squeeze(-1).T


class Predictor:
    def __init__(self, model, preprocessing, config):
        self.model, self.preprocessing, self.config = model, preprocessing, config

    def __call__(self, states):
        p = self.preprocessing
        x = (features(states, p["molecular_weights"], self.config["input_floor"])-p["x_offset"])/p["x_scale"]
        if self.config["local"]:
            output = self.model(x @ p["basis"])
        else:
            with torch.no_grad():
                output = np.concatenate([self.model(torch.as_tensor(chunk, dtype=torch.float32)).double().numpy()
                    for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/512))))])
        coordinate = output*p["y_scale"] + p["y_offset"]
        value = 1e-14*np.sinh(coordinate)
        value[:, ~p["active"]] = 0.
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite Arrhenius-input model prediction")
        return value, np.zeros_like(value, dtype=bool)


def prepare(training, molecular_weights, config, active):
    x = features(training["states"], molecular_weights, config["input_floor"])
    offset, scale = standardization(x)
    target = np.arcsinh(training["delta"]/1e-14)
    y_offset, y_scale = standardization(target)
    p = dict(x_offset=offset, x_scale=scale, y_offset=y_offset, y_scale=y_scale,
             molecular_weights=np.asarray(molecular_weights), active=np.asarray(active))
    return (x-offset)/scale, (target-y_offset)/y_scale, p


def fit(training, validation, physics, config, active, destination):
    wall, cpu = time.monotonic(), time.process_time()
    x, target, p = prepare(training, physics["molecular_weights"], config, active)
    if config["local"]:
        _, singular, vh = np.linalg.svd(x, full_matrices=False)
        basis = vh[singular > singular[0]*1e-10].T
        p.update(basis=basis, points=x @ basis, targets=target)
        if basis.shape[1]+1 >= min(config["neighbors"], len(x)):
            raise ValueError("Too few local neighbors for polynomial rank")
        model = RBFInterpolator(p["points"], target, neighbors=min(config["neighbors"], len(x)),
                                smoothing=config["smoothing"], kernel="cubic", degree=1)
        history, step, count = [], 0, None
    else:
        model = SpeciesHeads(x.shape[1], target.shape[1], config["widths"], config["seed"])
        x, target = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)
        reference = torch.as_tensor(training["delta"])
        mask = torch.as_tensor(active)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        rng, history = np.random.default_rng(config["seed"]), []
        predictor = Predictor(model, p, config)
        use_lbfgs = config["name"] == "arrhenius-lbfgs"
        adam_updates = config["adam_updates"] if use_lbfgs else config["updates"]
        half = adam_updates if use_lbfgs else config["updates"]//2
        for step in range(1, adam_updates+1):
            if time.monotonic()-wall > config["wall_seconds"]:
                raise TimeoutError("Arrhenius-input fit exceeded declared wall limit")
            physical = step > half
            if step == half+1:
                optimizer = torch.optim.Adam(model.parameters(), lr=config["physical_learning_rate"])
            first = config["physical_learning_rate"] if physical else config["learning_rate"]
            last = config["physical_final_learning_rate"] if physical else config["final_learning_rate"]
            progress = ((step-half if physical else step)-1)/max(half-1, 1)
            optimizer.param_groups[0]["lr"] = last + .5*(first-last)*(1+np.cos(np.pi*progress))
            indices = rng.integers(0, len(x), config["batch_size"])
            output = model(x[indices])
            if physical:
                z = output.double()*torch.as_tensor(p["y_scale"]) + torch.as_tensor(p["y_offset"])
                prediction = 1e-14*torch.sinh(z)
                loss = physical_loss(prediction, reference[indices], mask, reference[indices], 0., 0.)
            else:
                loss = (output[:, mask]-target[indices][:, mask]).abs().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite Arrhenius-input loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step % config["validation_every"] == 0 or step == adam_updates:
                record = dict(step=step, phase="physical" if physical else "coordinate", batch_loss=float(loss.detach()))
                for name, rows in (("training_probe", {key: value[:256] for key, value in training.items()}), ("validation", validation)):
                    pred, _ = predictor(rows["states"])
                    record[name] = tolerance_counts(pred[:, active], rows["delta"][:, active], 1e-15, .1)
                history.append(record)
        if use_lbfgs:
            torch.save(model.state_dict(), destination / "warmup-weights.pt")
            optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1,
                history_size=config["lbfgs_history"], line_search_fn="strong_wolfe",
                tolerance_grad=1e-12, tolerance_change=1e-15)
            evaluations = 0

            def closure():
                nonlocal evaluations
                if time.monotonic()-wall > config["wall_seconds"] or evaluations >= config["lbfgs_max_evaluations"]:
                    raise TimeoutError("Bounded L-BFGS evaluation budget reached")
                evaluations += 1
                optimizer.zero_grad(set_to_none=True)
                output = model(x).double()
                z = output*torch.as_tensor(p["y_scale"]) + torch.as_tensor(p["y_offset"])
                ratio = (1e-14*torch.sinh(z[:, mask])-reference[:, mask])/(1e-15+.1*reference[:, mask].abs())
                # Each independent species contributes its RMS normalized error.
                # The tiny smoothing constant keeps gradients finite at exact fit.
                loss = torch.sqrt(ratio.square().mean(dim=0)+1e-12).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite L-BFGS objective; no silent clipping")
                loss.backward()
                if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                    raise ValueError("Nonfinite L-BFGS gradient")
                return loss

            for iteration in range(1, config["lbfgs_steps"]+1):
                loss = optimizer.step(closure)
                step = adam_updates+iteration
                if iteration % 10 == 0 or iteration == config["lbfgs_steps"]:
                    record = dict(step=step, phase="full-batch-lbfgs", batch_loss=float(loss.detach()),
                                  closure_evaluations=evaluations)
                    for name, rows in (("training_probe", {key: value[:256] for key, value in training.items()}), ("validation", validation)):
                        pred, _ = predictor(rows["states"])
                        record[name] = tolerance_counts(pred[:, active], rows["delta"][:, active], 1e-15, .1)
                    history.append(record)
        torch.save(model.state_dict(), destination / "weights.pt")
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        count = sum(v.numel() for v in model.parameters())
    np.savez_compressed(destination / "arrhenius-preprocessing.npz", **p)
    return Predictor(model, p, config), dict(history=history, updates_completed=step, parameter_count=count,
        optimizer_evaluations=evaluations if config["name"] == "arrhenius-lbfgs" else None,
        table_bytes=sum(a.nbytes for a in p.values()) if config["local"] else None,
        fit_process_seconds=time.process_time()-cpu, fit_wall_seconds=time.monotonic()-wall)


def reload(directory, config):
    with np.load(directory / "arrhenius-preprocessing.npz", allow_pickle=False) as arrays:
        p = {key: arrays[key] for key in arrays.files}
    if config["local"]:
        model = RBFInterpolator(p["points"], p["targets"], neighbors=min(config["neighbors"], len(p["points"])),
                                smoothing=config["smoothing"], kernel="cubic", degree=1)
    else:
        model = SpeciesHeads(len(p["x_offset"]), len(p["active"]), config["widths"], config["seed"])
        model.load_state_dict(torch.load(directory / "weights.pt", weights_only=True, map_location="cpu"))
        model.eval()
    return Predictor(model, p, config)
