"""Trusted species-distance controller. Default is read-only discovery."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,digest,checked_file,verify_evaluator_lock,project_path
from benchmarks.research_contract.replay import inputs
from benchmarks.research_contract.admission import acquire_lease,process_identity
from benchmarks.research_contract.sandbox import probe
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.campaign import processes_on_gpu,run_worker
from ..hidden_features.prepare import save
from ..local_predictability.neighbors import verify_cpu
from .metric import calibration_mask,rank_weights,scale_features
from .analysis import score_transfers

EXPERIMENT = "EXP-20261013-species-distance"


def check_prior(project,plan):
    root = project/plan["prior_run"]
    review = read_json(checked_file(root/"review.json",plan["prior_review_sha256"]))
    report = read_json(checked_file(root/"report.json",review["report_sha256"]))
    if review["status"] != "verified":
        raise ValueError("Prior diagnostic must be independently verified")
    for path,value in report["artifacts"].items():
        checked_file(project_path(root,path),value)
    return root


def prepare(prior,output,seed,plan,active):
    source,staged = prior/str(seed),output/str(seed)
    (staged/"input").mkdir(parents=True)
    library,query = arrays(source/"library.npz"),arrays(source/"queries.npz")
    features = arrays(source/"input/features.npz")
    for population,count in plan["query_counts"].items():
        if int((query["population"]==population).sum()) != count:
            raise ValueError("Query population changed")
    training = query["population"] == "training_loo"
    mask = calibration_mask(library["source_indices"],query["source_indices"][training])
    if len(library["states"]) != 50000 or mask.sum() != 48976:
        raise ValueError("Metric calibration population changed")
    unseen_ids = features["query_ids"][~training]
    if np.isin(library["source_indices"],unseen_ids).any():
        raise ValueError("Evaluation row entered metric/library")
    target = (np.arcsinh(library["delta"][mask][:,active]/1e-14)
              -np.arcsinh(library["base"][mask][:,active]/1e-14))
    weights,rho = rank_weights(library["features"][mask],target,plan["uniform_weight_fraction"])
    np.savez(staged/"calibration.npz",weights=weights,correlations=rho,
             source_indices=library["source_indices"][mask])
    np.savez(staged/"input/features.npz",**features,weights=weights)
    save(staged/"input/plan.json",plan)
    for name in ("queries.npz","library.npz","selection.json"):
        shutil.copyfile(source/name,staged/name)
    save(staged/"calibration.json",dict(rows=int(mask.sum()),excluded_training_queries=int(training.sum()),
         query_labels_used=False,weight_min=float(weights.min()),weight_max=float(weights.max()),
         per_species_effective_features=((weights.sum(axis=1)**2)/(weights**2).sum(axis=1)).tolist()))
    return staged,library,query,features,weights


def check_graphs(prior,staged,features,weights,seed,plan):
    graph = arrays(staged/"worker/neighbors.npz")
    old = arrays(prior/str(seed)/"worker/neighbors.npz")
    graphs,checks = {},[]
    for size in plan["libraries"]:
        global_ix = graph[f"global_{size}_indices"]
        np.testing.assert_array_equal(global_ix,old[f"correction_{size}_indices"])
        np.testing.assert_array_equal(graph[f"global_{size}_distance"],old[f"correction_{size}_distance"])
        graphs["global",size] = np.repeat(global_ix[:,None,:],58,axis=1)
        all_indices = []
        for species,weight in enumerate(weights):
            key = f"species_{species}_{size}"
            indices,distance = graph[key+"_indices"],graph[key+"_distance"]
            if (indices.shape != (4095,8) or np.any(indices<0) or np.any(indices>=size)
                    or np.any(features["query_ids"][:,None] == features["library_ids"][indices])):
                raise ValueError("Invalid or self-matching neighbor")
            count = verify_cpu(scale_features(features["library_features"],weight),
                               scale_features(features["query_features"],weight),
                               features["library_ids"],features["query_ids"],
                               dict(indices=indices,distance=distance),size,
                               sample=plan["cpu_verification_queries_per_species_library"])
            checks.append(dict(seed=seed,library=size,species_index=species,cpu_queries=count,
                               nearest_differs_from_global=float((indices[:,0]!=global_ix[:,0]).mean())))
            all_indices.append(indices)
        graphs["species_rank",size] = np.stack(all_indices,axis=1)
    return graphs,checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("New output required; reconcile active work before retry")
    project = args.project.resolve()
    binding = read_json(project/"research/bindings.json")
    plan = read_json(project/"research/experiments"/(EXPERIMENT+".json"))
    lock = verify_evaluator_lock(project)
    inputs(project,binding)
    prior = check_prior(project,plan)
    if source_revision()["dirty"] or not read_json(project/"research/campaign.json")["execution_authorized"]:
        raise ValueError("Require clean committed source and authority")
    if processes_on_gpu(args.gpu):
        print(json.dumps(dict(status="WAITING_RESOURCE")))
        return
    print(json.dumps(dict(status="PRECHECKED",execute=args.execute)),flush=True)
    if not args.execute:
        return
    start = time.monotonic()
    args.output.mkdir(parents=True)
    acquire_lease(project/"runs/research-control"/(EXPERIMENT+".lease.json"),
                  dict(experiment=EXPERIMENT,output=str(args.output.resolve()),start_ticks=process_identity(os.getpid())))
    save(args.output/"plan.json",plan)
    save(args.output/"source.json",source_revision())
    save(args.output/"evaluator-lock.json",lock)
    for name in ("probe-input","probe-output"):
        (args.output/name).mkdir()
    save(args.output/"isolation.json",json.loads(probe(project,args.output/"probe-input",args.output/"probe-output",args.gpu)))
    records,checks,timings = [],[],[]
    state = args.output/"active-job.json"
    active = np.array([name != "AR" for name in binding["species"]])
    for seed in plan["seeds"]:
        begin = time.monotonic()
        staged,library,query,features,weights = prepare(prior,args.output,seed,plan,active)
        preparation = time.monotonic()-begin
        worker = run_worker(project,staged/"input",staged/"worker",[],state,args.gpu,
                            worker_module="benchmarks.offline_accuracy.paper_baseline.species_distance.worker")
        begin = time.monotonic()
        graphs,verified = check_graphs(prior,staged,features,weights,seed,plan)
        records.extend(score_transfers(staged,library,query,graphs,binding["species"],seed))
        checks.extend(verified)
        timings.append(dict(seed=seed,preparation_seconds=preparation,worker_seconds=worker,
                            verification_and_scoring_seconds=time.monotonic()-begin))
        print(json.dumps(dict(seed=seed,status="SCORED")),flush=True)
    verify_evaluator_lock(project)
    save(state,dict(status="COMPLETE_DIAGNOSTIC",candidate_promoted=False))
    save(args.output/"report.json",dict(status="complete",records=records,checks=checks,timings=timings,
         elapsed_seconds=time.monotonic()-start,neural_fits=0,candidate_promoted=False,
         artifacts={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file()}))
    save(args.output/"verification.json",dict(report_sha256=digest(args.output/"report.json")))


if __name__ == "__main__":
    main()
