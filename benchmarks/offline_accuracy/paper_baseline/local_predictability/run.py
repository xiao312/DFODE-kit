"""Trusted diagnostic controller; default is read-only provenance discovery."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, digest, checked_file, verify_evaluator_lock
from benchmarks.research_contract.replay import inputs, population_rows
from benchmarks.research_contract.admission import acquire_lease, process_identity
from benchmarks.research_contract.sandbox import probe
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.campaign import processes_on_gpu, run_worker
from ..hidden_features.prepare import save
from .analysis import stratified_rows, transfer, summarize
from .neighbors import verify_cpu

EXPERIMENT = "EXP-20261013-local-predictability"


def prepare(project, output, binding, seed, plan):
    root = project / binding["prepared_root"]
    source = project / "runs/offline-accuracy/diversity-20261013"
    checked_file(source / "report.json", read_json(source / "verification.json")["report_sha256"])
    report = read_json(source / "report.json")
    relative = f"prepared/{seed}/training-indices.npy"
    indices = np.load(checked_file(source / relative, report["artifacts"][relative]), allow_pickle=False)
    full = arrays(root / str(seed) / "training.npz")
    library = {k:v[indices] for k,v in full.items()}
    del full
    queries, names, positions = [], [], {}
    for name, rows in population_rows(root, project / binding["transfer_root"], str(seed)):
        if name == "training":
            selected = stratified_rows(rows,256)
            name = "training_loo"
        elif name == "same_source_unseen":
            if np.intersect1d(rows["source_indices"], library["source_indices"]).size:
                raise ValueError("Unseen evaluation rows entered library")
            selected = stratified_rows(rows,512)
        else:
            if set(rows["snapshot"]) & set(library["snapshot"]):
                raise ValueError("Development snapshot entered training library")
            selected = np.arange(len(rows["states"]))
        positions[name] = selected.tolist()
        queries.append({k:v[selected] for k,v in rows.items()})
        names.extend([name]*len(selected))
        if len(selected) != plan["expected_query_counts"][name]:
            raise ValueError("Unexpected diagnostic query count")
    combined = {k:np.concatenate([q[k] for q in queries]) for k in queries[0]}
    populations = np.array(names)
    qualified_ids = combined["source_indices"].copy()
    if np.any(qualified_ids<0) or np.any(library["source_indices"]<0):
        raise ValueError("ID namespace mapping requires nonnegative source IDs")
    dev = populations == "development"
    qualified_ids[dev] = -1-qualified_ids[dev]
    staged = output / str(seed)
    (staged / "input").mkdir(parents=True)
    np.savez(staged / "input/features.npz", library_features=library["features"],
             library_ids=library["source_indices"], query_features=combined["features"], query_ids=qualified_ids)
    save(staged / "input/plan.json", plan)
    # Trusted artifacts remain outside the worker's sole input mount.
    np.savez(staged / "queries.npz", **combined, population=populations)
    np.savez(staged / "library.npz", **library)
    save(staged / "selection.json", dict(seed=seed, query_positions=positions,
         library_positions=indices.tolist(), self_exclusion="dataset namespace plus source ID",
         library_hash=digest(staged / "library.npz"), query_hash=digest(staged / "queries.npz")))
    return staged, library, combined, populations, qualified_ids


def analyze(staged, library, query, populations, query_ids, species, seed, plan):
    graph = arrays(staged / "worker/neighbors.npz")
    active = np.array([name != "AR" for name in species])
    records, graphs = [], []
    for space, width in plan["spaces"].items():
        for size in plan["libraries"]:
            key = f"{space}_{size}"
            ix, distance = graph[key+"_indices"], graph[key+"_distance"]
            if np.any(query_ids[:,None] == library["source_indices"][ix]):
                raise ValueError("Self row was not excluded")
            checks = verify_cpu(library["features"][:,:width], query["features"][:,:width],
                                library["source_indices"], query_ids,
                                dict(indices=ix,distance=distance), size)
            permutation = np.random.default_rng(seed+17).permutation(size)
            for population in np.unique(populations):
                mask = populations == population
                graphs.append(dict(seed=seed, space=space, library=size, population=population,
                    query_states=int(mask.sum()), checked_cpu_queries=checks,
                    nearest_distance_quantiles=np.quantile(distance[mask,0],[0,.5,.95,.99,1]).tolist(),
                    zero_distance_queries=int((distance[mask,0]==0).sum()),
                    same_snapshot_fraction=float((query["snapshot"][mask]==library["snapshot"][ix[mask,0]]).mean())))
            for k, shuffled in ((1,False),(8,False),(1,True)):
                selected = permutation[ix[:,:k]] if shuffled else ix[:,:k]
                for mode in ("target","correction"):
                    pred, coordinate = transfer(library, query["base"], selected, active, mode)
                    name = f"{key}-k{k}-{mode}" + ("-shuffled" if shuffled else "")
                    np.savez(staged / (name+".npz"), prediction=pred, coordinate=coordinate,
                             source_indices=query["source_indices"], population=populations)
                    for population in np.unique(populations):
                        mask = populations == population
                        q = {key:value[mask] for key,value in query.items()}
                        metric = summarize(pred[mask],coordinate[mask],q,active,species,mode)
                        record = dict(seed=seed,space=space,library=size,k=k,mode=mode,
                                      shuffled=shuffled,population=population,artifact=name+".npz",score=metric)
                        records.append(record)
                        print(json.dumps({key:record[key] for key in ("seed","space","library","k","mode","shuffled","population")} | dict(acceptance=metric["groups"]["all"]["acceptance"])), flush=True)
    return records, graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("New output required; reconcile existing work rather than repeat it")
    project = args.project.resolve()
    binding = read_json(project / "research/bindings.json")
    lock = verify_evaluator_lock(project)
    plan = read_json(project / "research/experiments" / (EXPERIMENT+".json"))
    campaign = read_json(project / "research/campaign.json")
    inputs(project,binding)
    if source_revision()["dirty"] or not campaign["execution_authorized"]:
        raise ValueError("Require clean committed source and existing execution authority")
    if processes_on_gpu(args.gpu):
        print(json.dumps(dict(status="WAITING_RESOURCE")))
        return
    print(json.dumps(dict(status="PRECHECKED",execute=args.execute)),flush=True)
    if not args.execute:
        return
    started = time.monotonic()
    args.output.mkdir(parents=True)
    acquire_lease(project / "runs/research-control" / (EXPERIMENT+".lease.json"),
                  dict(experiment=EXPERIMENT,output=str(args.output.resolve()),start_ticks=process_identity(os.getpid())))
    save(args.output / "plan.json",plan)
    save(args.output / "source.json",source_revision())
    save(args.output / "evaluator-lock.json",lock)
    for folder in ("probe-input","probe-output"):
        (args.output / folder).mkdir()
    save(args.output / "isolation.json",json.loads(probe(project,args.output / "probe-input",args.output / "probe-output",args.gpu)))
    records, graphs = [], []
    timings = []
    state = args.output / "active-job.json"
    for seed in plan["seeds"]:
        begin = time.monotonic()
        staged,library,query,populations,ids = prepare(project,args.output,binding,seed,plan)
        preparation = time.monotonic()-begin
        worker_wall = run_worker(project,staged / "input",staged / "worker",[],state,args.gpu,
                                worker_module="benchmarks.offline_accuracy.paper_baseline.local_predictability.neighbors")
        begin = time.monotonic()
        r,g = analyze(staged,library,query,populations,ids,binding["species"],seed,plan)
        records.extend(r); graphs.extend(g)
        timings.append(dict(seed=seed,preparation_seconds=preparation,worker_seconds=worker_wall,
                            scoring_seconds=time.monotonic()-begin))
        del library,query
    verify_evaluator_lock(project)
    save(state,dict(status="COMPLETE_DIAGNOSTIC",candidate_promoted=False))
    save(args.output / "report.json",dict(status="complete",records=records,graphs=graphs,timings=timings,
         elapsed_seconds=time.monotonic()-started,neural_fits=0,candidate_promoted=False,
         scope="Local transfer diagnostic only; not candidate selection or independent evaluation",
         artifacts={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file()}))
    save(args.output / "verification.json",dict(report_sha256=digest(args.output / "report.json")))


if __name__ == "__main__":
    main()
