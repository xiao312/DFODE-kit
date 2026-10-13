"""Trusted common training selection and label-free inference staging."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json
from benchmarks.research_contract.replay import population_rows
from ..paired_base.metrics import align_indices
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker


def stage(project,dest,seed,plan):
    binding=read_json(project/"research/bindings.json");root=project/binding["prepared_root"];transfer=project/binding["transfer_root"]
    selection=arrays(transfer/"plan/selections.npz");full={};excluded=[]
    for s in plan["seeds"]:
        with np.load(root/str(s)/"training.npz") as z:
            full[s]={k:z[k] for k in ("states","delta","base","source_indices","snapshot")}
        excluded.extend(full[s]["source_indices"][np.r_[np.arange(10000),selection[f"unseen_{s}"]]])
    common=np.intersect1d(full[plan["seeds"][0]]["source_indices"],full[plan["seeds"][1]]["source_indices"])
    available=np.setdiff1d(common,np.unique(excluded))
    ids=np.random.default_rng(plan["selection_seed"]).permutation(available)[:plan["fit_rows"]]
    if len(ids)!=plan["fit_rows"]:raise ValueError("Insufficient common fitting support")
    own={k:v[align_indices(full[seed]["source_indices"],ids)] for k,v in full[seed].items()}
    other=next(s for s in plan["seeds"] if s!=seed);other_ix=align_indices(full[other]["source_indices"],ids)
    for key in ("states","delta","snapshot"):np.testing.assert_array_equal(own[key],full[other][key][other_ix])
    teacher=full[other]["base"][other_ix];del full
    model=dest/"model";model.mkdir()
    for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(root/str(seed)/name,model/name)
    fit=dest/"fit-input";fit.mkdir();np.savez(fit/"training.npz",**own,other=teacher);save(fit/"config.json",plan)
    np.savez(dest/"training-reference.npz",**own)
    for population,rows in [("training",own),*list(population_rows(root,transfer,str(seed)))[1:]]:
        inputs=dest/("input-"+population);inputs.mkdir();np.savez(inputs/"states.npz",states=rows["states"],source_indices=rows["source_indices"]);save(inputs/"config.json",plan)
        if population!="training":np.savez(dest/(population+"-reference.npz"),**rows)
    save(dest/"config.json",plan)


def execute(project,dest,output,seed,state,gpu):
    elapsed=run_worker(project,dest/"fit-input",output,["--seed",str(seed)],state,gpu,dest/"model",
        worker_module="benchmarks.offline_accuracy.paper_baseline.consensus_distill.worker")
    model=dest/"inference-model";shutil.copytree(dest/"model",model)
    for name in ("weights.pt","config.json","constant.npy"):shutil.copyfile(output/name,model/name)
    for population in ("training","same_source_unseen","development"):
        elapsed+=run_worker(project,dest/("input-"+population),output/population,["--seed",str(seed)],state,gpu,model,
            worker_module="benchmarks.offline_accuracy.paper_baseline.consensus_distill.infer")
    return elapsed
