"""One bounded fit, with physical stopping and preservation of the selected state."""
import copy
import json
import time

import numpy as np
import torch

from train import network, preprocessing
from fit_diagnostic import physical_scores, state_hash


def budget_max(predicted, reference, weights):
    error = np.abs(predicted - reference) / weights
    if not np.all(np.isfinite(error)):
        raise ValueError("Budget errors must be finite")
    return float(error.max())


def fit(data, indices, config, method, directory, deadline):
    if method not in ("adam-decay", "lbfgs", "linear-head"):
        raise ValueError("Unknown polish method")
    if not np.all(data["split"] == "train"):
        raise ValueError("Training rows only")
    started = time.monotonic()
    deadline = min(deadline, started + config["fit_seconds"])
    input_norm, output_norm, _, active, encoded = preprocessing(data, "budget-linear")
    x = torch.as_tensor(input_norm.transform(data["inputs"][indices]), dtype=torch.float64)
    target = torch.as_tensor(output_norm.transform(encoded[indices]), dtype=torch.float64)
    active_tensor = torch.as_tensor(active)
    model = network(x.shape[1], target.shape[1], config["hidden_widths"], config["seed"], torch.float64)
    result = {"method": method, "seed": config["seed"], "rows": len(indices),
              "row_ids": data["id"][indices].tolist(), "initial_weights_sha256": state_hash(model),
              "history": [], "closure_evaluations": 0, "stop_reason": "update_limit"}
    best_max, best_state, selected = float("inf"), None, None

    def check_time():
        if time.monotonic() > deadline:
            raise TimeoutError("Bounded fit wall limit")

    def prediction():
        with torch.no_grad():
            values = model(x).numpy()
        values[:, ~active] = 0
        return values * output_norm.scale * data["weights"][indices]

    def measure(stage, step):
        nonlocal best_max, best_state, selected
        predicted = prediction()
        error = np.abs(predicted - data["delta"][indices]) / data["weights"][indices]
        maximum = budget_max(predicted, data["delta"][indices], data["weights"][indices])
        point = {"stage": stage, "step": step, "max_budget": maximum,
                 "species_max": float(error[:, 1:].max()), "temperature_max": float(error[:, 0].max())}
        result["history"].append(point)
        if maximum < best_max:
            best_max, best_state, selected = maximum, copy.deepcopy(model.state_dict()), point.copy()
        return maximum <= config["stop_budget_max"]

    def loss():
        value = (model(x)[:, active_tensor] - target[:, active_tensor]).square().mean()
        if not torch.isfinite(value):
            raise ValueError("Loss must be finite")
        return value

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    passed = measure("initial", 0)
    try:
        for step in range(1, config["warmup_updates"] + 1):
            if passed:
                break
            check_time()
            optimizer.zero_grad(set_to_none=True)
            loss().backward()
            optimizer.step()
            passed = measure("warmup", step)
        result["warmup_weights_sha256"] = state_hash(model)
        if not passed and method == "linear-head":
            check_time()
            with torch.no_grad():
                hidden = model[:-1](x).numpy()
            features = np.column_stack([hidden, np.ones(len(hidden))])
            coefficients, _, rank, singular = np.linalg.lstsq(features, target.numpy(), rcond=None)
            result["feature_rank"] = int(rank)
            result["feature_condition"] = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
            with torch.no_grad():
                model[-1].weight.copy_(torch.as_tensor(coefficients[:-1].T))
                model[-1].bias.copy_(torch.as_tensor(coefficients[-1]))
            passed = measure("linear-head", 1)
        elif not passed and method == "adam-decay":
            for step in range(1, config["adam_updates"] + 1):
                check_time()
                fraction = (step - 1) / max(config["adam_updates"] - 1, 1)
                optimizer.param_groups[0]["lr"] = config["learning_rate"] * (config["adam_final_rate"] / config["learning_rate"])**fraction
                optimizer.zero_grad(set_to_none=True)
                loss().backward()
                optimizer.step()
                passed = measure("adam-decay", step)
                if passed:
                    break
        elif not passed and method == "lbfgs":
            optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=25,
                                         history_size=config["lbfgs_history"], tolerance_grad=0.,
                                         tolerance_change=0., line_search_fn="strong_wolfe")

            def closure():
                check_time()
                result["closure_evaluations"] += 1
                optimizer.zero_grad(set_to_none=True)
                value = loss()
                value.backward()
                return value

            for step in range(1, config["lbfgs_steps"] + 1):
                check_time()
                optimizer.step(closure)
                # Trial line-search points are not accepted states and cannot pass.
                passed = measure("lbfgs", step)
                if passed:
                    break
        if passed:
            result["stop_reason"] = "budget_pass"
    except TimeoutError:
        result["stop_reason"] = "time_limit"
    except (ValueError, RuntimeError) as error:
        result["stop_reason"] = "numerical_failure"
        result["error"] = str(error)
    model.load_state_dict(best_state)
    predicted = prediction()
    result["selected"] = selected
    result["final"] = physical_scores(predicted, data, indices)
    result["elapsed_seconds"] = time.monotonic() - started
    result["selected_weights_sha256"] = state_hash(model)
    directory.mkdir()
    torch.save(model.state_dict(), directory / "weights.pt")
    np.savez(directory / "predictions.npz", predicted=predicted, indices=indices)
    np.savez(directory / "preprocessing.npz", input_offset=input_norm.offset, input_scale=input_norm.scale,
             output_scale=output_norm.scale, active=active)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
