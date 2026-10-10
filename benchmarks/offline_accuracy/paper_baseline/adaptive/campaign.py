"""Preview a finite six-fit GPU queue; stop on any failed run or pairing check."""
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
from .plan import ARMS,SEEDS,configuration,BASE_HASHES
from .gpu import inventory,idle


def commands(args):
    stages = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f"{seed}-{arm}"
            command = [sys.executable,"-m","benchmarks.offline_accuracy.paper_baseline.adaptive.run"]
            for key in ("campaign","previous","original","base-root","baseline"):
                command.extend(["--"+key,str(getattr(args,key.replace("-","_")))])
            for key,value in (("output",args.output/name),("arm",arm),("seed",seed)):
                command.extend(["--"+key,str(value)])
            stages.append((name,command+["--execute"]))
    return stages


def check_result(result):
    config = result["config"]
    if config != configuration(config["arm"],config["seed"]):
        raise ValueError("Adaptive configuration changed")
    expected = dict(status="complete",training_count=200000,development_count=1023,independent_test_count=0,
                    updates_completed=6000,row_presentations=60000000,zero_correction_identity=True,
                    baseline_result_sha256=BASE_HASHES[config["seed"]])
    if any(result.get(key) != value for key,value in expected.items()):
        raise ValueError("Adaptive result has changed work, data or baseline identity")
    if [row["updates"] for row in result["history"]] != list(range(1000,6001,1000)):
        raise ValueError("Incomplete adaptive history")
    if result["residual_arithmetic"]["max_error_over_allowance"] > .01:
        raise ValueError("Residual arithmetic failed")


def verify_completed(root,names):
    from ..matched_work.run import checked_result
    results = {}
    for name in names:
        path = root/name
        result = checked_result(path)
        check_result(result)
        seed,arm = result["config"]["seed"],result["config"]["arm"]
        if name != f"{seed}-{arm}" or name in results:
            raise ValueError("Unexpected run identity")
        checks = json.loads((path/"verification.json").read_text())
        for key in ("exact_model_replay","independent_paired_counts","physical_checks","train_only_calibration","frozen_base"):
            if checks.get(key) is not True:
                raise ValueError("Missing adaptive verification: "+key)
        first_name = f"{seed}-continue"
        if arm != "continue":
            first = results[first_name]
            for key in ("hashes","source","frozen_base_weights_sha256","baseline_result_sha256"):
                if first[key] != result[key]:
                    raise ValueError("Unpaired adaptive source or base: "+key)
            for filename in ("training-indices.npy","normalization-indices.npy"):
                np.testing.assert_array_equal(np.load(root/first_name/filename),np.load(path/filename))
            with np.load(root/first_name/"preprocessing.npz") as a,np.load(path/"preprocessing.npz") as b:
                if set(a.files) != set(b.files):
                    raise ValueError("Preprocessing schema differs")
                for key in a.files:
                    np.testing.assert_array_equal(a[key],b[key])
        if arm == "calibrated":
            local = results[f"{seed}-local"]
            for key in ("initial_weights_sha256","parameter_count","deployment_parameter_count"):
                if result[key] != local[key]:
                    raise ValueError("Correction arms differ beyond scale: "+key)
        results[name] = result
    return [dict(name=name,result_sha256=sha256(root/name/"result.json")) for name in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","baseline","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    stages = commands(args)
    print(json.dumps(stages),flush=True)
    if not args.execute:
        return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        parser.error("Select an idle GPU and deterministic CUDA workspace")
    args.output.mkdir(parents=True)
    status = dict(status="running",pid=os.getpid(),started_unix=time.time(),stages=[])
    save = lambda: save_json(args.output/"campaign-status.json",status)
    save()
    completed_names = []
    try:
        for name,command in stages:
            selected = os.environ["CUDA_VISIBLE_DEVICES"]
            if not any(row["index"] == selected and idle(row) for row in inventory()):
                raise RuntimeError("Selected GPU is no longer idle; no next fit started")
            stage = dict(name=name,status="running",started_unix=time.time(),command=command)
            status["stages"].append(stage)
            save()
            with (args.output/(name+".log")).open("x") as log:
                completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1500)
            stage.update(exit_code=completed.returncode,ended_unix=time.time())
            if completed.returncode:
                raise RuntimeError("Fit failed: "+name+"; inspect its log")
            completed_names.append(name)
            verify_completed(args.output,completed_names)
            stage["status"] = "verified"
            save()
        if completed_names != [f"{seed}-{arm}" for seed in SEEDS for arm in ARMS]:
            raise ValueError("Incomplete adaptive matrix")
        evidence = verify_completed(args.output,completed_names)
        save_json(args.output/"verification.json",dict(status="verified",common_inputs=True,
            frozen_base=True,paired_correction_initialization=True,results=evidence))
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed",error=str(error))
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1]["status"] = "failed"
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
