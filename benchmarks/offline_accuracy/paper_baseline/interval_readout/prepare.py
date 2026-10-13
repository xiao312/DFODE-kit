"""Pinned common-row staging; state-only inference mounts remain separate."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker


def stage(project,dest,seed,plan):
    binding=read_json(project/'research/bindings.json'); prepared=project/binding['prepared_root']
    source=project/plan['parent_run']/'prepared'/str(seed)
    rows=arrays(source/'training-reference.npz')
    shutil.copyfile(source/'training-reference.npz',dest/'training-reference.npz')
    model=dest/'model';model.mkdir()
    for name in ('base-weights.pt','preprocessing.npz'):shutil.copyfile(prepared/str(seed)/name,model/name)
    fit=dest/'fit-input';fit.mkdir();np.savez(fit/'training.npz',**rows);save(fit/'config.json',plan)
    for population,items in [('training',rows),*list(population_rows(prepared,project/binding['transfer_root'],str(seed)))[1:]]:
        inputs=dest/('input-'+population);inputs.mkdir()
        np.savez(inputs/'states.npz',states=items['states'],source_indices=items['source_indices']);save(inputs/'config.json',plan)
        if population!='training':np.savez(dest/(population+'-reference.npz'),**items)


def execute(project,dest,output,seed,state,gpu):
    elapsed=run_worker(project,dest/'fit-input',output,['--seed',str(seed)],state,gpu,dest/'model',
                       worker_module='benchmarks.offline_accuracy.paper_baseline.interval_readout.worker')
    model=dest/'inference-model';shutil.copytree(dest/'model',model)
    shutil.copyfile(output/'scales.npz',model/'scales.npz')
    for arm in ('point','interval'):
        (model/arm).mkdir()
        for name in ('weights.pt','config.json'):shutil.copyfile(output/arm/name,model/arm/name)
    for population in ('training','same_source_unseen','development'):
        elapsed+=run_worker(project,dest/('input-'+population),output/population,['--seed',str(seed)],state,gpu,model,
                           worker_module='benchmarks.offline_accuracy.paper_baseline.interval_readout.infer')
    return elapsed
