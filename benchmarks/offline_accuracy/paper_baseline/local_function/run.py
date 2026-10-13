"""Matched local-model controller using the existing isolation and scorer."""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,digest,checked_file,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,population_rows
from benchmarks.research_contract.admission import acquire_lease,process_identity,validate
from benchmarks.research_contract.sandbox import probe
from benchmarks.research_contract.gates import decision
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save,verify_stage
from ..hidden_features.campaign import run_worker,processes_on_gpu,score_arm
from .model import select_centres


def stage(project,output,binding,plan):
    if plan.get("model_kind")=="base_polish":
        from ..base_polish.prepare import stage as polish_stage
        return polish_stage(project,output,plan)
    if plan.get("model_kind")=="constant_calibration":
        from ..interval_calibration.prepare import stage as calibration_stage
        return calibration_stage(project,output,plan)
    if plan.get("parent_run"):
        parent=project/plan["parent_run"]
        review=read_json(checked_file(parent/"review.json",plan["parent_review_sha256"]))
        report=read_json(checked_file(parent/"report.json",review["report_sha256"]))
        for relative,expected in report["artifacts"].items():
            if relative.startswith("prepared/"):checked_file(parent/relative,expected)
        shutil.copytree(parent/"prepared",output/"prepared")
        for seed in plan["seeds"]:
            prep=output/"prepared"/str(seed)
            for arm in plan["arms"]:
                path=prep/("train-"+arm)
                save(path/"config.json",dict(plan,seed=seed,arm=arm))
                for name,dest in [("geometry.npz","geometry.npz"),("weights.pt","initial-weights.pt")]:
                    relative=f"fits/{seed}-{arm}/model/{name}"
                    shutil.copyfile(checked_file(parent/relative,report["artifacts"][relative]),path/dest)
            metadata=read_json(prep/"manifest.json")
            metadata["artifacts"]={str(p.relative_to(prep)):digest(p) for p in prep.rglob("*") if p.is_file() and p.name!="manifest.json"}
            save(prep/"manifest.json",metadata)
        return
    prior=project/"runs/offline-accuracy/species-distance-20261013"
    review=read_json(checked_file(prior/"review.json","fa5caa1b06632a7406fea11d3edb91f51f81cd43bb071eb2c443ae4a1adfc249"))
    report=read_json(checked_file(prior/"report.json",review["report_sha256"]))
    for seed in plan["seeds"]:
        prefix=str(seed)+"/"
        for relative,expected in report["artifacts"].items():
            if relative in [prefix+"library.npz",prefix+"calibration.npz",prefix+"selection.json"]:
                checked_file(prior/relative,expected)
        library=arrays(prior/str(seed)/"library.npz")
        calibration=arrays(prior/str(seed)/"calibration.npz")
        weights=calibration["weights"]
        np.testing.assert_allclose(weights.mean(axis=1),1,atol=1e-14,rtol=0)
        if weights.shape!=(58,119) or np.any(weights<.1):raise ValueError("Metric identity/normalization failed")
        centres=select_centres(library["source_indices"],library["snapshot"],seed,plan["centres"])
        prepared=output/"prepared"/str(seed);(prepared/"model").mkdir(parents=True)
        source=project/binding["prepared_root"]/str(seed)
        active=arrays(source/"preprocessing.npz")["active"]
        for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(source/name,prepared/"model"/name)
        indices=np.array(read_json(prior/str(seed)/"selection.json")["library_positions"])
        np.save(prepared/"training-indices.npy",indices)
        for arm in plan["arms"]:
            path=prepared/("train-"+arm);path.mkdir()
            cfg=dict(plan,seed=seed,arm=arm)
            np.savez(path/"training.npz",features=library["features"],target=library["target"],
                     delta=library["delta"],base=library["base"],active=active,
                     source_indices=library["source_indices"],centre_indices=centres,
                     weights=weights if arm=="species" else np.ones_like(weights))
            save(path/"config.json",cfg)
        for population,rows in population_rows(project/binding["prepared_root"],project/binding["transfer_root"],str(seed)):
            if population=="training":rows=library
            if population=="same_source_unseen" and np.isin(rows["source_indices"],library["source_indices"]).any():
                raise ValueError("Unseen row entered fitting")
            path=prepared/("input-"+population);path.mkdir()
            np.savez(path/"states.npz",states=rows["states"],source_indices=rows["source_indices"])
        save(prepared/"manifest.json",dict(training_states=50000,metric_calibration_rows=48976,
             centres=library["source_indices"][centres].tolist(),unseen_overlap=0,
             artifacts={str(p.relative_to(prepared)):digest(p) for p in prepared.rglob("*") if p.is_file()}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--experiment",default="EXP-20261013-local-function")
    p.add_argument("--gpu",type=int,default=0);p.add_argument("--execute",action="store_true")
    args=p.parse_args();project=args.project.resolve()
    if args.output.exists():p.error("Output exists: reconcile rather than duplicate")
    plan=read_json(project/"research/experiments"/(args.experiment+".json"))
    binding=read_json(project/"research/bindings.json");campaign=read_json(project/"research/campaign.json")
    lock=verify_evaluator_lock(project);inputs(project,binding)
    if source_revision()["dirty"]:raise ValueError("Commit before execution")
    if processes_on_gpu(args.gpu):print("WAITING_RESOURCE");return
    if not campaign["execution_authorized"]:raise ValueError("Missing authority")
    print(json.dumps(dict(status="PRECHECKED",execute=args.execute)),flush=True)
    if not args.execute:return
    args.output.mkdir(parents=True)
    acquire_lease(project/"runs/research-control"/(args.experiment+".lease.json"),
                  dict(experiment=args.experiment,output=str(args.output.resolve()),start_ticks=process_identity(os.getpid())))
    save(args.output/"source.json",source_revision());save(args.output/"frozen-experiment.json",plan)
    save(args.output/"evaluator-lock.json",lock)
    for name in ("probe-input","probe-output"):(args.output/name).mkdir()
    save(args.output/"isolation.json",json.loads(probe(project,args.output/"probe-input",args.output/"probe-output",args.gpu)))
    m0=project/"runs/offline-accuracy/research-contract-20261013/m0"
    m0report=read_json(checked_file(m0/"report.json",read_json(m0/"verification.json")["report_sha256"]))
    validate(campaign,{"budget":plan},dict(metric=campaign["metric"],m0_verified=m0report["status"]=="M0_SCORE_PARITY_PASSED",
        identities_verified=True,inference_isolation_verified=True,source_clean=True,idle_gpu=True),[])
    stage(project,args.output,binding,plan)
    state=args.output/"active-job.json";records=[];elapsed=0
    calibration=plan.get("model_kind")=="constant_calibration"
    polish=plan.get("model_kind")=="base_polish"
    worker="benchmarks.offline_accuracy.paper_baseline."+("base_polish" if polish else "interval_calibration" if calibration else "local_function")+".worker"
    for arm in plan["arms"]:
        for seed in plan["seeds"]:
            prep=args.output/"prepared"/str(seed);verify_stage(prep)
            dest=args.output/"fits"/f"{seed}-{arm}"
            extra=["--arm",arm] if calibration or polish else []
            elapsed+=run_worker(project,prep/("train-"+arm),dest/"fit",["train","--seed",str(seed),*extra],state,args.gpu,worker_module=worker)
            model=dest/"model";shutil.copytree(prep/"model",model)
            for name in (("weights.pt",) if polish else ("constants.npz",) if calibration else ("weights.pt","geometry.npz")):shutil.copyfile(dest/"fit"/name,model/name)
            for population in ("training","same_source_unseen","development"):
                elapsed+=run_worker(project,prep/("input-"+population),dest/population,["infer","--seed",str(seed),*extra],state,args.gpu,model,worker_module=worker)
            record=score_arm(project,args.output,binding,seed,arm,campaign);records.append(record)
            save(dest/"score.json",record);verify_evaluator_lock(project)
            print(json.dumps(dict(arm=arm,seed=seed,acceptance={k:v["candidate"]["groups"]["all"]["acceptance"] for k,v in record["populations"].items()})),flush=True)
    expected=[f"{s}:{p}:{c}" for s in plan["seeds"] for p in ("same_source_unseen","development") for c in ("anchor","incumbent")]
    decisions={}
    for arm in plan["arms"]:
        selected=[r for r in records if r["arm"]==arm]
        decisions[arm]=decision({k:v for r in selected for k,v in r["gates"].items()},expected,
                               max(v for r in selected for v in r["latency_ratios"]),campaign["screening"])
    save(state,dict(status="COMPLETE_AWAITING_INTERNAL_RECOUNT",decisions=decisions))
    save(args.output/"report.json",dict(status="complete",records=records,decisions=decisions,worker_wall_seconds=elapsed,
         artifacts={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file()}))
    save(args.output/"verification.json",dict(report_sha256=digest(args.output/"report.json")))


if __name__=="__main__":main()
