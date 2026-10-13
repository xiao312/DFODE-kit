"""Registered training-only diagnostic under the existing resource/isolation guard."""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.admission import acquire_lease,process_identity,validate
from benchmarks.research_contract.sandbox import probe
from benchmarks.research_contract.replay import inputs
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker,processes_on_gpu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0);p.add_argument("--execute",action="store_true")
    args=p.parse_args();project=args.project.resolve();output=args.output.resolve()
    plan=read_json(project/"research/experiments/EXP-20261013-gradient-probe.json")
    campaign=read_json(project/"research/campaign.json");lock=verify_evaluator_lock(project)
    inputs(project,read_json(project/"research/bindings.json"))
    if output.exists():raise ValueError("Existing output: reconcile; do not duplicate")
    if source_revision()["dirty"]:raise ValueError("Commit before execution")
    if processes_on_gpu(args.gpu):print("WAITING_RESOURCE");return
    if not campaign["execution_authorized"]:raise ValueError("Missing authorization")
    parent=project/plan["parent_run"]
    review=read_json(checked_file(parent/"review.json",plan["parent_review_sha256"]))
    report=read_json(checked_file(parent/"report.json",review["report_sha256"]))
    print(json.dumps(dict(status="PRECHECKED",execute=args.execute)),flush=True)
    if not args.execute:return
    output.mkdir(parents=True)
    acquire_lease(project/"runs/research-control"/(plan["experiment_id"]+".lease.json"),dict(output=str(output),start_ticks=process_identity(os.getpid())))
    save(output/"source.json",source_revision());save(output/"frozen-experiment.json",plan);save(output/"evaluator-lock.json",lock)
    for name in ("probe-input","probe-output"):(output/name).mkdir()
    save(output/"isolation.json",json.loads(probe(project,output/"probe-input",output/"probe-output",args.gpu)))
    m0=project/"runs/offline-accuracy/research-contract-20261013/m0"
    m0report=read_json(checked_file(m0/"report.json",read_json(m0/"verification.json")["report_sha256"]))
    validate(campaign,{"budget":dict(updates=plan["batches"],batch_size=plan["batch_size"])},dict(metric=campaign["metric"],m0_verified=m0report["status"]=="M0_SCORE_PARITY_PASSED",identities_verified=True,inference_isolation_verified=True,source_clean=True,idle_gpu=True),[])
    records=[];elapsed=0
    for seed in plan["seeds"]:
        prefix=f"prepared/{seed}/"
        for relative,expected in report["artifacts"].items():
            if relative.startswith(prefix):checked_file(parent/relative,expected)
        source=parent/"prepared"/str(seed);dest=output/"prepared"/str(seed);dest.mkdir(parents=True)
        training=arrays(source/"train-global/training.npz");states=arrays(source/"input-training/states.npz")
        np.testing.assert_array_equal(training["source_indices"],states["source_indices"])
        selected=np.random.default_rng(seed+137).permutation(len(states["states"]))[:plan["batches"]*plan["batch_size"]]
        np.savez(dest/"training.npz",states=states["states"][selected],delta=training["delta"][selected],base=training["base"][selected],source_indices=training["source_indices"][selected])
        save(dest/"config.json",plan);shutil.copytree(source/"model",dest/"model")
        elapsed+=run_worker(project,dest,output/str(seed),["--seed",str(seed)],output/"active-job.json",args.gpu,dest/"model",worker_module="benchmarks.offline_accuracy.paper_baseline.gradient_probe.worker")
        records.append(read_json(output/str(seed)/"result.json"));verify_evaluator_lock(project)
        print(json.dumps(dict(seed=seed,status=records[-1]["status"])),flush=True)
    save(output/"report.json",dict(records=records,worker_wall_seconds=elapsed,artifacts={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}))
    save(output/"verification.json",dict(report_sha256=digest(output/"report.json")))
    save(output/"active-job.json",dict(status="COMPLETE_AWAITING_RECOUNT"))


if __name__=="__main__":main()
