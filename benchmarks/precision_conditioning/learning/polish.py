"""Run a pinned multi-seed polishing test using existing training data only."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fit_diagnostic import training_data, subsets
from polish_core import fit
from pilot import revision


def configuration():
    config = json.loads(Path(__file__).with_name("polish_experiment.json").read_text())
    if len(config["seeds"]) != 3 or len(set(config["seeds"])) != 3:
        raise ValueError("Three distinct seeds are required")
    if config["methods"] != ["adam-decay", "lbfgs", "linear-head"]:
        raise ValueError("Expected the three prescribed methods")
    for key in ("adam_updates", "lbfgs_steps", "lbfgs_history", "fit_seconds", "wall_time_seconds"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < config["stop_budget_max"] <= .5 or not 0 < config["adam_final_rate"] < config["learning_rate"]:
        raise ValueError("Invalid stop margin or learning-rate decay")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = configuration()
    datasets = {name: training_data(args.run / "training" / name / "dataset.npz") for name in ("h2", "ch4")}
    chosen = {name: subsets(data, name, config) for name, data in datasets.items()}
    print(json.dumps({"fits": 54, "seeds": config["seeds"], "methods": config["methods"],
                      "subsets": {name: {key: len(rows) for key, rows in selections.items()} for name, selections in chosen.items()},
                      "test_rows_used": 0, "config": config}), flush=True)
    if args.dry_run:
        return
    if torch.__version__ != "2.5.1+cpu":
        raise ValueError("Expected torch 2.5.1+cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config["wall_time_seconds"]
    summary = {"status": "running", "source": revision(), "torch": torch.__version__, "config": config,
               "input_run": args.run.name, "test_rows_used": 0, "datasets": {}, "results": []}

    def save():
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))

    for name, data in datasets.items():
        directory = args.output / name
        directory.mkdir()
        np.savez(directory / "dataset.npz", **data)
        summary["datasets"][name] = {"training_rows": len(data["id"]), "input_sha256": hashlib.sha256(
            (args.run / "training" / name / "dataset.npz").read_bytes()).hexdigest()}
    save()
    try:
        # Finish the narrow milestone across mechanisms/seeds before the all-row bridge.
        for subset in ("one", "narrow", "all"):
            for name, data in datasets.items():
                for seed in config["seeds"]:
                    for method in config["methods"]:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Global run wall limit")
                        relative = f"{name}/{subset}-{seed}-{method}"
                        result = fit(data, chosen[name][subset], dict(config, seed=seed), method, args.output / relative, deadline)
                        result.update(mechanism=name, subset=subset, directory=relative)
                        summary["results"].append(result)
                        save()
                        print(json.dumps({"fit": relative, "stop": result["stop_reason"],
                                          "max_budget": result["selected"]["max_budget"],
                                          "selected": result["selected"]["step"]}), flush=True)
        summary["status"] = "complete"
    except TimeoutError:
        summary["status"] = "time_limit"
    except Exception as error:
        summary["status"], summary["error"] = "failed", str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
