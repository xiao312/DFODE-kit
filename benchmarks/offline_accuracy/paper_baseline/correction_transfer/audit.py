"""Four bounded CPU workers, preserving references separately from training labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cantera as ct
import numpy as np
import scipy
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from .prepare import checked_plan


def run_one(index,plan,mechanism,deadline):
    if time.monotonic()>=deadline:
        return dict(index=index,status="not_started",error="Overall audit deadline")
    env = dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    command = [sys.executable,"-m","benchmarks.offline_accuracy.paper_baseline.correction_transfer.audit_worker",
               str(plan),str(mechanism),str(index)]
    try:
        result = subprocess.run(command,capture_output=True,text=True,timeout=min(120,deadline-time.monotonic()),env=env)
        if result.returncode:
            return dict(index=index,status="failed",error=result.stderr[-4000:])
        record = json.loads(result.stdout)
        record["stderr"] = result.stderr
        return record
    except subprocess.TimeoutExpired:
        return dict(index=index,status="timeout",error="Reference worker exceeded fixed wall limit")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan","dataset","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Reference output must be new")
    plan = checked_plan(args.plan)
    mechanism = args.dataset/"mechanism.yaml"
    if sha256(args.dataset/"manifest.json")!=plan["dataset_manifest_sha256"] or sha256(mechanism)!=plan["mechanism_sha256"]:
        raise ValueError("Reference mechanism provenance differs")
    gas = ct.Solution(str(mechanism))
    if gas.species_names!=plan["species"] or plan["interval_s"]!=1e-6:
        raise ValueError("Unexpected chemistry interface")
    selected = json.loads((args.plan/"audit-selection.json").read_text())
    if len(selected)!=80 or len({r["source_row"] for r in selected})!=80:
        raise ValueError("Require the 80 distinct predeclared reference states")
    print(json.dumps(dict(status="planned",states=80,cpu_workers=4,wall_seconds=1200)),flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit reference source first")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    report = dict(status="running",source=source,plan_sha256=sha256(args.plan/"plan.json"),
        versions=dict(python=sys.version,cantera=ct.__version__,scipy=scipy.__version__,numpy=np.__version__),
        cpu_workers=4,records=[])
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_one,i,args.plan,mechanism,started+1200) for i in range(80)]
        for future in as_completed(futures):
            record = future.result()
            save(args.output/f"record-{record['index']:03d}.json",record)
            compact = {k:v for k,v in record.items() if k not in ("checks","state","stderr")}
            report["records"].append(compact)
            report["elapsed_seconds"] = time.monotonic()-started
            save(args.output/"report.json",report)
            print(json.dumps(dict(completed=len(report["records"]),index=record["index"],status=record["status"])),flush=True)
    report["records"].sort(key=lambda r:r["index"])
    report["status"] = "complete" if all(r["status"]=="complete" for r in report["records"]) else "incomplete"
    report["artifacts"] = {q.name:sha256(q) for q in args.output.glob("record-*.json")}
    save(args.output/"report.json",report)
    save(args.output/"verification.json",dict(report_sha256=sha256(args.output/"report.json")))


if __name__=="__main__":
    main()
