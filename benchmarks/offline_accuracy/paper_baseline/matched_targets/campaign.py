"""Preview or run a finite sequential queue and validate the complete paired matrix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.parallel_labels.storage import save_json
from .plan import TARGETS, OBJECTIVES, SEEDS
from .checks import check_result, check_pair, check_arrays, matrix_names


def commands(args):
    stages = []
    for seed in SEEDS:
        for target in TARGETS:
            for objective in OBJECTIVES:
                name = f"{seed}-{target}-{objective}"
                command = [sys.executable, "-m", "benchmarks.offline_accuracy.paper_baseline.matched_targets.run"]
                for key in ("campaign", "previous", "original", "base-root", "baseline"):
                    command.extend(["--"+key, str(getattr(args, key.replace("-", "_")))])
                for key, value in (("output", args.output / name), ("target", target), ("objective", objective), ("seed", seed)):
                    command.extend(["--"+key, str(value)])
                stages.append((name, command+["--execute"]))
    return stages


def verify_completed(root, names):
    from ..matched_work.run import checked_result
    results = {}
    for name in names:
        result = checked_result(root / name)
        check_result(result)
        config = result["config"]
        seed, target = config["seed"], config["target"]
        expected = f"{seed}-{target}-{config['objective']}"
        if expected != name or name in results:
            raise ValueError("Unexpected or repeated result identity")
        first_name = f"{seed}-state-boxcox-coordinate"
        if name != first_name:
            check_pair(results[first_name], result, same_target=False)
            check_arrays(root / first_name, root / name, same_target=False)
        coordinate_name = f"{seed}-{target}-coordinate"
        if name != coordinate_name:
            check_pair(results[coordinate_name], result, same_target=True)
            check_arrays(root / coordinate_name, root / name, same_target=True)
        if name == first_name and result.get("exact_historical_control_parity") is not True:
            raise ValueError("Missing exact historical control check")
        results[name] = result
    return [dict(name=name, result_sha256=sha256(root / name / "result.json")) for name in results]


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
    args.output.mkdir(parents=True)
    status = dict(status="running", pid=os.getpid(), started_unix=time.time(), stages=[])
    save = lambda: save_json(args.output / "campaign-status.json", status)
    save()
    completed_names = []
    try:
        for name, command in stages:
            stage = dict(name=name, status="running", started_unix=time.time(), command=command)
            status["stages"].append(stage)
            save()
            with (args.output / (name+".log")).open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1500)
            stage.update(exit_code=completed.returncode, ended_unix=time.time())
            if completed.returncode:
                stage["status"] = "failed"
                raise RuntimeError(f"Fit failed: {name}; inspect its log")
            completed_names.append(name)
            verify_completed(args.output, completed_names)
            stage["status"] = "verified"
            save()
        if completed_names != matrix_names():
            raise ValueError("Incomplete experiment matrix")
        evidence = verify_completed(args.output, completed_names)
        save_json(args.output / "verification.json", dict(status="verified", paired_warmup=True,
            common_inputs=True, common_initial_weights=True, historical_state_control_parity=True, results=evidence))
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed", error=str(error))
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1]["status"] = "failed"
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
