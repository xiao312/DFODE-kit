"""Bounded idle-GPU wait, focused GPU tests, then the finite adaptive campaign."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmarks.flame_conditioning.parallel_labels.storage import save_json
from .campaign import commands
from .gpu import inventory,idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","baseline","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    stages = commands(args)
    print(json.dumps(dict(gpus=inventory(),stages=stages)),flush=True)
    if not args.execute:
        return
    status_path = args.output.parent/(args.output.name+"-launcher-status.json")
    if status_path.exists():
        parser.error("Launcher already exists; inspect it before choosing a new output")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    status = dict(status="waiting-for-idle-gpu",started_unix=time.time(),pid=os.getpid())
    save = lambda: save_json(status_path,status)
    save()
    try:
        selected = None
        while time.time()-status["started_unix"] < 21600:
            rows = inventory()
            status.update(gpus=rows,checked_unix=time.time())
            save()
            candidate = next((r for r in rows if idle(r)),None)
            if candidate:
                time.sleep(5)
                if any(r["uuid"] == candidate["uuid"] and idle(r) for r in inventory()):
                    selected = candidate
                    break
            time.sleep(60)
        if selected is None:
            raise TimeoutError("No idle GPU within six hours; no training started")
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=selected["index"],CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
        status.update(status="testing",gpu=selected)
        save()
        subprocess.run([sys.executable,"-c","from benchmarks.offline_accuracy.paired.runtime import configure; configure()"],
            env=env,check=True,timeout=60)
        subprocess.run([sys.executable,"-m","pytest","tests/test_adaptive_representation.py",
            "tests/test_matched_targets.py","-q","-p","no:cacheprovider"],env=env,check=True,timeout=180)
        subprocess.run(stages[0][1][:-1],env=env,check=True,timeout=300)
        status["status"] = "running-campaign"
        save()
        command = [sys.executable,"-m","benchmarks.offline_accuracy.paper_baseline.adaptive.campaign"]
        for key in ("campaign","previous","original","base-root","baseline","output"):
            command.extend(["--"+key,str(getattr(args,key.replace("-","_")))])
        subprocess.run(command+["--execute"],env=env,check=True,timeout=9600)
        status["status"] = "complete"
    except Exception as error:
        status.update(status="failed",error=str(error))
        raise
    finally:
        status["ended_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
