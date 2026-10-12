"""Finite eight-fit queue; retain failed evidence and stop rather than retune."""
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
from ..adaptive.gpu import inventory,idle
from .plan import ARMS,SEEDS,BASE_HASHES,configuration


def commands(args):
    stages = []
    for seed in SEEDS:
        for arm in ARMS:
            name = f"{seed}-{arm}"
            cmd = [sys.executable,"-m","benchmarks.offline_accuracy.paper_baseline.decoder_loss.run"]
            for key in ("campaign","previous","original","base-root","baseline"):
                cmd.extend(["--"+key,str(getattr(args,key.replace("-","_")))])
            for key,value in (("output",args.output/name),("arm",arm),("seed",seed)):
                cmd.extend(["--"+key,str(value)])
            stages.append((name,cmd+["--execute"]))
    return stages


def verify_completed(root,names):
    from ..matched_work.run import checked_result
    results = {}
    for name in names:
        path = root/name
        r = checked_result(path)
        cfg = r["config"]
        if cfg!=configuration(cfg["arm"],cfg["seed"]) or name!=f"{cfg['seed']}-{cfg['arm']}" or name in results:
            raise ValueError("Unrecognized recipe/result identity")
        expected = dict(training_count=200000,development_count=1023,independent_test_count=0,
            updates_completed=6000,row_presentations=60000000,gradient_clipping_count=0,zero_correction_identity=True,
            baseline_result_sha256=BASE_HASHES[cfg["seed"]])
        if any(r.get(k)!=v for k,v in expected.items()):
            raise ValueError("Changed data, work, baseline or clipping")
        checks = json.loads((path/"verification.json").read_text())
        if any(checks.get(key) is not True for key in ("exact_model_replay","independent_paired_counts","physical_checks",
                "train_only_preprocessing","frozen_base","baseline_prediction_replay")):
            raise ValueError("Missing verification")
        if [row["updates"] for row in r["history"]]!=list(range(1000,6001,1000)):
            raise ValueError("Incomplete history")
        with (path/"gradients.jsonl").open() as handle:
            gradients = [json.loads(line) for line in handle]
        if (len(gradients)!=6000 or any(row["update"]!=i+1 or row["clipped"] or
                not np.isfinite(row["raw_l2"]) for i,row in enumerate(gradients))):
            raise ValueError("Missing or invalid gradient evidence")
        first_name = f"{cfg['seed']}-linear-log"
        if name!=first_name:
            first = results[first_name]
            for key in ("hashes","source","initial_weights_sha256","frozen_base_weights_sha256","parameter_count"):
                if r[key]!=first[key]:
                    raise ValueError("Unpaired comparison: "+key)
            for filename in ("training-indices.npy","normalization-indices.npy"):
                np.testing.assert_array_equal(np.load(path/filename),np.load(root/first_name/filename))
            with np.load(path/"preprocessing.npz") as a,np.load(root/first_name/"preprocessing.npz") as b:
                if set(a.files)!=set(b.files):
                    raise ValueError("Preprocessing keys differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key],b[key])
            if r["artifacts"]["decoder-audit.json"]!=first["artifacts"]["decoder-audit.json"]:
                raise ValueError("Common decoder audit differs")
        results[name] = r
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
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not selected or os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":
        parser.error("Select an idle GPU and deterministic CUDA workspace")
    args.output.mkdir(parents=True)
    status = dict(status="running",pid=os.getpid(),started_unix=time.time(),stages=[])
    save = lambda:save_json(args.output/"campaign-status.json",status)
    save()
    names = []
    try:
        for name,cmd in stages:
            # Allow utilization meters to settle after this queue's previous process exits.
            for attempt in range(13):
                if any(row["index"]==selected and idle(row) for row in inventory()):
                    break
                if attempt==12:
                    raise RuntimeError("Selected GPU remained busy; no next fit started")
                time.sleep(5)
            stage = dict(name=name,status="running",started_unix=time.time(),command=cmd)
            status["stages"].append(stage)
            save()
            with (args.output/(name+".log")).open("x") as log:
                completed = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            stage.update(exit_code=completed.returncode,ended_unix=time.time())
            if completed.returncode:
                raise RuntimeError("Failed fit: "+name)
            names.append(name)
            verify_completed(args.output,names)
            stage["status"] = "verified"
            save()
        if names!=[f"{s}-{a}" for s in SEEDS for a in ARMS]:
            raise ValueError("Incomplete matrix")
        save_json(args.output/"verification.json",dict(status="verified",common_inputs=True,
            paired_initialization=True,results=verify_completed(args.output,names)))
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed",error=str(error))
        if status["stages"] and status["stages"][-1]["status"]=="running":
            status["stages"][-1]["status"] = "failed"
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__=="__main__":
    main()
