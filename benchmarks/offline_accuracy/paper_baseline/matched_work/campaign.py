"""Preview or execute eight sequential, equally budgeted GPU fits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.parallel_labels.storage import save_json
from .plan import configuration, update_policy
from .run import checked_result


def commands(campaign, previous, original, base_root, output):
    stages = []
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            for count in (50000, 200000):
                name = f"{seed}-{recipe}-n{count}"
                command = [sys.executable, "-m", "benchmarks.offline_accuracy.paper_baseline.matched_work.run"]
                for key, value in (("campaign", campaign), ("previous", previous), ("original", original),
                                   ("base-root", base_root), ("output", output / name), ("recipe", recipe),
                                   ("seed", seed), ("training-count", count)):
                    command.extend(["--"+key, str(value)])
                stages.append((name, command+["--execute"]))
    return stages


def verify_campaign(root):
    evidence, initialization = [], {}
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            pair = []
            for count in (50000, 200000):
                path = root / f"{seed}-{recipe}-n{count}"
                result = checked_result(path)
                config = configuration(recipe, seed)
                if (result["config"] != config or result["updates_completed"] != config["updates"] or
                        result["row_presentations"] != config["updates"]*config["batch_size"] or
                        result["effective_batch_size"] != config["batch_size"] or result["training_count"] != count):
                    raise ValueError("Work budget or experiment config differs")
                expected = []
                last = None
                for completed in range(config["updates"]):
                    rate, reset = update_policy(config, completed)
                    if rate != last:
                        expected.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
                        last = rate
                if result["schedule"] != expected:
                    raise ValueError("Learning-rate schedule differs")
                init = result["initial_weights_sha256"]
                if seed in initialization and initialization[seed] != init:
                    raise ValueError("Initial weights differ for paired seed")
                initialization[seed] = init
                pair.append((path, result))
                evidence.append(dict(name=path.name, result_sha256=sha256(path / "result.json")))
            (small_path, small), (large_path, large) = pair
            for key in ("preprocessing_array_sha256", "hashes", "development_count"):
                if small[key] != large[key]:
                    raise ValueError("Pair source or preprocessing differs")
            with np.load(small_path / "development-predictions.npz") as a, np.load(large_path / "development-predictions.npz") as b:
                np.testing.assert_array_equal(a["source_indices"], b["source_indices"])
            small_ids = np.load(small_path / "training-indices.npy", allow_pickle=False)
            large_ids = np.load(large_path / "training-indices.npy", allow_pickle=False)
            np.testing.assert_array_equal(small_ids, large_ids[:50000])
            for path in (small_path, large_path):
                np.testing.assert_array_equal(np.load(path / "normalization-indices.npy", allow_pickle=False), small_ids)
    return dict(status="verified", equal_work=True, common_normalization=True,
                nested_selections=True, identical_initial_weights=True, results=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "previous", "original", "base-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new campaign output")
    stages = commands(args.campaign, args.previous, args.original, args.base_root, args.output)
    print(json.dumps(stages), flush=True)
    if not args.execute:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("Select an idle GPU and set CUBLAS_WORKSPACE_CONFIG=:4096:8")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    args.output.mkdir(parents=True)
    status = dict(status="running", pid=os.getpid(), started_unix=time.time(), stages=[])
    def save():
        save_json(args.output / "campaign-status.json", status)
    save()
    try:
        for name, command in stages:
            stage = dict(name=name, status="running", started_unix=time.time(), command=command)
            status["stages"].append(stage)
            save()
            with (args.output / (name+".log")).open("x") as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1500)
            stage.update(exit_code=completed.returncode, ended_unix=time.time(),
                         status="complete" if completed.returncode == 0 else "failed")
            save()
            if completed.returncode:
                raise RuntimeError(f"Stage {name} failed; inspect its log. Later fits were not started.")
        save_json(args.output / "verification.json", verify_campaign(args.output))
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
