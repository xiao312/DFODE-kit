"""Preview or execute four fresh 18k-update fits on the frozen 200k pool."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.parallel_labels.storage import save_json
from .campaign import verify_campaign
from .budget_checks import check_pair
from .run import checked_result


def commands(args):
    stages = []
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            name = f"{seed}-{recipe}-u18000"
            command = [sys.executable, "-m", "benchmarks.offline_accuracy.paper_baseline.matched_work.run"]
            for key, value in (("campaign", args.campaign), ("previous", args.previous), ("original", args.original),
                               ("base-root", args.base_root), ("output", args.output / name), ("recipe", recipe),
                               ("seed", seed), ("training-count", 200000),
                               ("budget-baseline", args.baseline / f"{seed}-{recipe}-n200000")):
                command.extend(["--"+key, str(value)])
            stages.append((name, command+["--execute"]))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "previous", "original", "base-root", "baseline", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    stages = commands(args)
    print(json.dumps(stages), flush=True)
    if not args.execute:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("Select an idle GPU and deterministic CUDA workspace")
    baseline_check = json.loads((args.baseline / "verification.json").read_text())
    if verify_campaign(args.baseline) != baseline_check:
        raise ValueError("Baseline campaign verification changed")
    args.output.mkdir(parents=True)
    status = dict(status="running", pid=os.getpid(), started_unix=time.time(), stages=[])
    evidence = []
    save = lambda: save_json(args.output / "campaign-status.json", status)
    save()
    try:
        for name, command in stages:
            stage = dict(name=name, status="running", started_unix=time.time(), command=command)
            status["stages"].append(stage)
            save()
            with (args.output / (name+".log")).open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1500)
            stage.update(exit_code=completed.returncode, ended_unix=time.time(),
                         status="complete" if completed.returncode == 0 else "failed")
            save()
            if completed.returncode:
                raise RuntimeError(f"Fit failed: {name}; inspect its log")
            result = checked_result(args.output / name)
            seed, recipe = result["config"]["seed"], result["config"]["recipe"]
            old_path = args.baseline / f"{seed}-{recipe}-n200000"
            check_pair(old_path, args.output / name, checked_result(old_path), result)
            evidence.append(dict(name=name, result_sha256=sha256(args.output / name / "result.json")))
        save_json(args.output / "verification.json", dict(status="verified", fixed_data=True,
            common_normalization=True, identical_initial_weights=True,
            baseline_verification_sha256=sha256(args.baseline / "verification.json"), results=evidence))
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
