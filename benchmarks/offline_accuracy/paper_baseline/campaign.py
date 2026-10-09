"""Dry-run by default; bounded reference preparation then four sequential GPU fits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def commands(source, original, base_root, output):
    python = sys.executable
    data, audit = output / "dataset", output / "audit"
    stages = [("prepare", 3700, [python, "-m", "benchmarks.flame_conditioning.prepare", str(source),
        "--config", "benchmarks/offline_accuracy/paper_baseline/dataset-50k.json", "--output", str(data)]),
        ("audit", 960, [python, "-m", "benchmarks.flame_conditioning.audit_labels", str(data), "--output", str(audit)])]
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            stages.append((f"{seed}-{recipe}", 1500, [python, "-m", "benchmarks.offline_accuracy.paper_baseline.run",
                str(data), "--audit", str(audit), "--base", str(base_root / f"seed-{seed}" / "training"),
                "--seed", str(seed), "--recipe", recipe, "--training-count", "50000",
                "--comparison-dataset", str(original), "--output", str(output / f"seed-{seed}" / recipe)]))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "base-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Campaign output must be new")
    stages = commands(args.source, args.original, args.base_root, args.output)
    print(json.dumps(stages), flush=True)
    if not args.execute:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("Select a free GPU and set CUBLAS_WORKSPACE_CONFIG=:4096:8")
    args.output.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    status = dict(status="running", pid=os.getpid(), stages=[], started_unix=time.time())
    path = args.output / "campaign-status.json"
    def save():
        path.write_text(json.dumps(status, indent=2, allow_nan=False))
    save()
    try:
        for name, timeout, command in stages:
            stage = dict(name=name, status="running", started_unix=time.time(), command=command)
            status["stages"].append(stage)
            save()
            with (args.output / (name+".log")).open("w") as log:
                process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            stage.update(exit_code=process.returncode, ended_unix=time.time(),
                         status="complete" if process.returncode == 0 else "failed")
            save()
            if process.returncode:
                raise RuntimeError(f"Stage {name} failed; inspect {name}.log. Later stages were not started.")
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
