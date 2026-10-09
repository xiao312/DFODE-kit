"""Dry-run by default; bounded reference preparation then four sequential GPU fits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def commands(source, original, base_root, output, training_count=50000, previous=None, workers=1):
    python = sys.executable
    data, audit = output / "dataset", output / "audit"
    stages = [("prepare", 3700, [python, "-m", "benchmarks.flame_conditioning.prepare", str(source),
        "--config", "benchmarks/offline_accuracy/paper_baseline/dataset-50k.json", "--output", str(data)]),
        ("audit", 960, [python, "-m", "benchmarks.flame_conditioning.audit_labels", str(data), "--output", str(audit)])]
    if training_count == 200000:
        if previous is None:
            raise ValueError("200k requires the complete 50k campaign")
        stages[0] = ("prepare", 3700, [python, "-m", "benchmarks.flame_conditioning.parallel_labels.run", str(source),
            "--config", "benchmarks/offline_accuracy/paper_baseline/dataset-200k.json", "--output", str(data),
            "--reuse", str(previous / "dataset"), "--workers", str(workers), "--execute"])
    elif training_count != 50000:
        raise ValueError("Select 50k or 200k")
    for seed in (20261011, 20261012):
        for recipe in ("fuel-state", "fuel-power"):
            stages.append((f"{seed}-{recipe}", 1500, [python, "-m", "benchmarks.offline_accuracy.paper_baseline.run",
                str(data), "--audit", str(audit), "--base", str(base_root / f"seed-{seed}" / "training"),
                "--seed", str(seed), "--recipe", recipe, "--training-count", str(training_count),
                "--comparison-dataset", str(original), "--output", str(output / f"seed-{seed}" / recipe)]))
            if training_count == 200000:
                stages[-1][2].extend(["--nested-run", str(previous / f"seed-{seed}" / recipe)])
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "original", "base-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--training-count", type=int, choices=(50000, 200000), default=50000)
    parser.add_argument("--previous-campaign", type=Path)
    parser.add_argument("--throughput-benchmark", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Campaign output must be new")
    stages = commands(args.source, args.original, args.base_root, args.output,
                      args.training_count, args.previous_campaign, args.workers)
    if args.training_count == 200000:
        from benchmarks.flame_conditioning.extract import sha256
        if args.throughput_benchmark is None:
            parser.error("200k requires a passing --throughput-benchmark")
        benchmark = json.loads(args.throughput_benchmark.read_text())
        measured = [row for row in benchmark["measurements"] if row["workers"] == args.workers]
        previous_status = json.loads((args.previous_campaign / "campaign-status.json").read_text())
        if (benchmark["status"] != "verified" or previous_status["status"] != "complete"
                or benchmark["dataset_manifest_sha256"] != sha256(args.previous_campaign / "dataset/manifest.json")
                or len(measured) != 2 or not all(row["exact_serial_labels"] and row["exact_acceptance_flags"] for row in measured)):
            parser.error("Require completed 50k campaign and two exact-parity worker measurements")
    print(json.dumps(stages), flush=True)
    if not args.execute:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("Select a free GPU and set CUBLAS_WORKSPACE_CONFIG=:4096:8")
    args.output.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    status = dict(status="running", pid=os.getpid(), stages=[], started_unix=time.time())
    if args.throughput_benchmark:
        from benchmarks.flame_conditioning.extract import sha256
        status["throughput_benchmark_sha256"] = sha256(args.throughput_benchmark)
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
