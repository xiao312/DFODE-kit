"""Reconcile a stopped registered run; never refit a completed worker result."""
import argparse
import os
from pathlib import Path
import shutil
import numpy as np
from benchmarks.research_contract.adapter import read_json, arrays, digest, verify_evaluator_lock, checked_file
from benchmarks.research_contract.admission import process_identity, acquire_lease
from benchmarks.research_contract.replay import inputs
from .prepare import verify_stage, save
from .model import config, ARMS
from .campaign import processes_on_gpu, run_worker, score_arm
from benchmarks.research_contract.gates import decision
from benchmarks.flame_conditioning.extract import source_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    verify_evaluator_lock(project)
    if source_revision()["dirty"]:
        raise ValueError("Commit controller recovery source first")
    if (args.run / "report.json").exists():
        checked_file(args.run / "report.json", read_json(args.run / "verification.json")["report_sha256"])
        print("Already complete; no work launched")
        return
    experiment = read_json(args.run / "frozen-experiment.json")
    lease = project / "runs/research-control" / (experiment["experiment_id"] + ".lease.json")
    owner = read_json(lease)
    if process_identity(owner["controller_pid"]) == owner["start_ticks"]:
        raise ValueError("Original controller still alive; attach instead")
    state = args.run / "active-job.json"
    current = read_json(state)
    if current.get("pid") and process_identity(current["pid"]) == current.get("start_ticks"):
        raise ValueError("Worker still alive; attach instead")
    if processes_on_gpu(args.gpu):
        print("WAITING_RESOURCE")
        return
    campaign, binding = [read_json(project / "research" / name) for name in ("campaign.json", "bindings.json")]
    if not campaign["execution_authorized"]:
        raise ValueError("Execution not authorized")
    inputs(project, binding)
    print("Stopped owner reconciled; completed fits will be reused", flush=True)
    if not args.execute:
        return
    acquire_lease(args.run / "recovery-lease.json", dict(start_ticks=process_identity(os.getpid()), source=source_revision()))
    save(args.run / "interruption.json", current)
    arms = [r["id"] for r in experiment["arms"]]
    records, elapsed, reused = [], float(current.get("elapsed_seconds", 0)), []
    for arm in arms:
        for seed in experiment["paired_seeds"]:
            prepared = args.run / "prepared" / str(seed)
            verify_stage(prepared)
            dest = args.run / "fits" / f"{seed}-{arm}"
            fit = dest / "fit"
            if fit.exists():
                completed = read_json(fit / "result.json")
                if (completed["status"] != "complete" or completed["config"] != config(seed,arm)
                        or completed["updates_completed"] != 6000 or completed["parameter_count"] != ARMS[arm][2]):
                    raise ValueError("Partial fit needs an explicit retry record, not silent reuse")
                reused.append(dict(seed=seed, arm=arm, result_sha256=digest(fit / "result.json"),
                                   weights_sha256=digest(fit / "weights.pt")))
            else:
                elapsed += run_worker(project, prepared / ("train-"+arm), fit,
                                      ["train","--arm",arm,"--seed",str(seed)],state,args.gpu)
            model = dest / "model"
            if not model.exists():
                shutil.copytree(prepared / "model",model)
                shutil.copyfile(fit / "weights.pt",model / "weights.pt")
            for population in ("training","same_source_unseen","development"):
                out = dest / population
                if out.exists():
                    result = read_json(out / "result.json")
                    checked_file(out / "predictions.npz",result["prediction_sha256"])
                else:
                    elapsed += run_worker(project,prepared / ("input-"+population),out,
                                          ["infer","--arm",arm,"--seed",str(seed)],state,args.gpu,model)
            record = score_arm(project,args.run,binding,seed,arm,campaign)
            records.append(record)
            save(dest / "score.json",record)
    expected = [f"{s}:{p}:{c}" for s in experiment["paired_seeds"] for p in ("same_source_unseen","development") for c in ("anchor","incumbent")]
    decisions = {}
    for arm in arms:
        selected = [r for r in records if r["arm"]==arm]
        decisions[arm] = decision({k:v for r in selected for k,v in r["gates"].items()},expected,
                                  max(v for r in selected for v in r["latency_ratios"]),campaign["screening"])
    save(args.run / "recovery.json",dict(source=source_revision(),reused_completed_fits=reused,
                                         duplicate_fits_launched=0,explanation="Resource monitor exit-race; validate/reuse completed fit, resume missing workers"))
    save(state,dict(status="REVIEW_REQUIRED",decisions={k:v["candidate_status"] for k,v in decisions.items()}))
    save(args.run / "report.json",dict(status="complete",decisions=decisions,records=records,worker_wall_seconds=elapsed,
        artifacts={str(p.relative_to(args.run)):digest(p) for p in args.run.rglob("*") if p.is_file()}))
    save(args.run / "verification.json",dict(report_sha256=digest(args.run / "report.json")))
    print({k:v["candidate_status"] for k,v in decisions.items()},flush=True)


if __name__ == "__main__":
    main()
