"""Reuse fixed role selection; stage minimal state-only inference artifacts."""
import shutil
from ..hidden_features.campaign import run_worker
from ..kinetic_features.prepare import stage
from benchmarks.research_contract.adapter import read_json


def execute(project,dest,output,seed,state,gpu):
    elapsed=run_worker(project,dest/"fit-input",output,["--seed",str(seed)],state,gpu,dest/"model",
        worker_module="benchmarks.offline_accuracy.paper_baseline.species_heads.worker")
    plan=read_json(dest/"config.json");model=dest/"inference-model";model.mkdir()
    for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(dest/"model"/name,model/name)
    for arm in plan["arms"]:
        (model/arm).mkdir()
        for name in ("weights.pt","config.json"):shutil.copyfile(output/arm/name,model/arm/name)
    return elapsed+run_worker(project,dest/"inference-input",output/"inference",["--seed",str(seed)],state,gpu,model,
        worker_module="benchmarks.offline_accuracy.paper_baseline.species_heads.infer")
