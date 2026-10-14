"""Bounded training-only diagnosis. No held-out model selection or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from train import network, preprocessing
from metrics import evaluate
from pilot import revision


def training_data(path):
    with np.load(path, allow_pickle=False) as saved:
        selected = saved["split"] == "train"
        data = {key: saved[key][selected] for key in saved.files}
    if not len(data["id"]) or len(set(data["id"])) != len(data["id"]):
        raise ValueError("Expected unique, nonempty training rows")
    return data


def subsets(data, mechanism, config):
    if not np.all(data["split"] == "train"):
        raise ValueError("Fit diagnosis must contain training rows only")
    parent = f"{mechanism}-{config['parent_temperature_K']}K"
    narrow = np.flatnonzero((data["parent"] == parent) &
                            np.isclose(data["inputs"][:, -1], np.log10(config["interval_s"]), atol=1e-12, rtol=0))
    if len(narrow) != config["narrow_rows"]:
        raise ValueError(f"Expected {config['narrow_rows']} accepted narrow rows for {mechanism}; got {len(narrow)}")
    return {"one": narrow[:1], "narrow": narrow, "all": np.arange(len(data["id"]))}


def objective(prediction, target, scale, active, kind):
    error = (prediction - target)[:, active]
    if kind == "physical-budget":
        error = error * (scale[active] / scale[active].max())
    elif kind != "coordinate":
        raise ValueError("Unknown loss")
    return error.square().mean()


def state_hash(model):
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()


def physical_scores(predicted, data, indices):
    result = evaluate(predicted, data, indices)
    result["temperature_budget_max"] = float(np.max(np.abs(predicted[:, 0] - data["delta"][indices, 0]) / data["weights"][indices, 0]))
    result["all_components_pass"] = bool(result["budget_max"] <= 1 and result["temperature_budget_max"] <= 1)
    return result


def run_fit(data, indices, config, kind, directory, deadline, head_only=False):
    started = time.monotonic()
    input_norm, output_norm, _, active, encoded = preprocessing(data, "budget-linear")
    x = torch.as_tensor(input_norm.transform(data["inputs"][indices]), dtype=torch.float64)
    y = torch.as_tensor(output_norm.transform(encoded[indices]), dtype=torch.float64)
    scale = torch.as_tensor(output_norm.scale, dtype=torch.float64)
    active_tensor = torch.as_tensor(active)
    model = network(x.shape[1], y.shape[1], config["hidden_widths"], config["seed"], torch.float64)
    result = {"loss": kind, "method": "linear-head-svd" if head_only else "adam",
              "rows": len(indices), "row_ids": data["id"][indices].tolist(),
              "initial_weights_sha256": state_hash(model), "updates": 0, "snapshots": [],
              "physical_loss_divisor": float(np.max(output_norm.scale[active])**2)}

    def predict():
        with torch.no_grad():
            q = model(x).numpy()
        q[:, ~active] = 0
        return q * output_norm.scale * data["weights"][indices]

    def snapshot(update):
        with torch.no_grad():
            q = model(x)
            losses = {name: objective(q, y, scale, active_tensor, name).item() for name in config["losses"]}
        return {"updates": update, "losses": losses, "scores": physical_scores(predict(), data, indices)}

    result["initial"] = snapshot(0)
    curves = []
    if head_only:
        with torch.no_grad():
            hidden = model[:-1](x).numpy()
        features = np.column_stack([hidden, np.ones(len(hidden))])
        coefficients, _, rank, singular = np.linalg.lstsq(features, y.numpy(), rcond=None)
        with torch.no_grad():
            model[-1].weight.copy_(torch.as_tensor(coefficients[:-1].T))
            model[-1].bias.copy_(torch.as_tensor(coefficients[-1]))
        result["feature_rank"] = int(rank)
        result["feature_condition"] = float(singular[0] / singular[-1])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        for update in range(1, config["updates"] + 1):
            if time.monotonic() > deadline:
                raise TimeoutError("Fit diagnosis reached its one-hour limit")
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), y, scale, active_tensor, kind)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite fit loss")
            loss.backward()
            optimizer.step()
            if update in config["snapshots"] or update == 1 or update % 100 == 0:
                measured = snapshot(update)
                curves.append({"updates": update, "losses": measured["losses"],
                               "species_p99": measured["scores"]["budget_p99"],
                               "species_max": measured["scores"]["budget_max"]})
                if update in config["snapshots"]:
                    result["snapshots"].append(measured)
        result["updates"] = config["updates"]
    prediction = predict()
    result["final"] = physical_scores(prediction, data, indices)
    result["elapsed_seconds"] = time.monotonic() - started
    result["curves"] = curves
    directory.mkdir()
    torch.save(model.state_dict(), directory / "weights.pt")
    np.savez(directory / "predictions.npz", predicted=prediction, indices=indices)
    np.savez(directory / "preprocessing.npz", input_offset=input_norm.offset, input_scale=input_norm.scale,
             output_scale=output_norm.scale, active=active)
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("fit_experiment.json").read_text())
    datasets = {name: training_data(args.run / "training" / name / "dataset.npz") for name in ("h2", "ch4")}
    selected = {name: subsets(data, name, config) for name, data in datasets.items()}
    plan = {"config": config, "subsets": {name: {key: len(rows) for key, rows in choices.items()} for name, choices in selected.items()},
            "adam_fits": 12, "linear_head_controls": 4, "test_rows_used": 0}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    if torch.__version__ != "2.5.1+cpu":
        raise RuntimeError("Expected torch 2.5.1+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"status": "running", "source": revision(), "torch": torch.__version__, "plan": plan,
               "input_run": args.run.name, "datasets": {}, "results": []}

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    save()
    try:
        for name, data in datasets.items():
            destination = args.output / name
            destination.mkdir()
            np.savez(destination / "dataset.npz", **data)
            _, normalizer, _, _, encoded = preprocessing(data, "budget-linear")
            roundtrip = normalizer.inverse(normalizer.transform(encoded)) * data["weights"]
            summary["datasets"][name] = {"input_sha256": hashlib.sha256((args.run / "training" / name / "dataset.npz").read_bytes()).hexdigest(),
                                          "roundtrip": physical_scores(roundtrip, data, np.arange(len(data["id"])))}
            for subset, indices in selected[name].items():
                for loss in config["losses"] + (["head-control"] if subset != "all" else []):
                    relative = f"{name}/{subset}-{loss}"
                    result = run_fit(data, indices, config, loss, args.output / relative,
                                     started + config["wall_time_seconds"], head_only=loss == "head-control")
                    result.update(mechanism=name, subset=subset, directory=relative)
                    summary["results"].append(result)
                    save()
                    print(json.dumps({"run": relative, "species_p99": result["final"]["budget_p99"],
                                      "all_components_pass": result["final"]["all_components_pass"]}), flush=True)
        summary["status"] = "complete"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
