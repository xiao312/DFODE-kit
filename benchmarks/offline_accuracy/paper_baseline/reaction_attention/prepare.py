"""Trusted subset staging and model-only/state-only inference mounts."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker


def stage(project, dest, seed, plan):
    binding=read_json(project/'research/bindings.json'); prepared=project/binding['prepared_root']
    source=project/plan['parent_run']/'prepared'/str(seed)
    full=arrays(source/'training-reference.npz')
    rows={k:v[:plan['fit_rows']] for k,v in full.items()}
    np.savez(dest/'training-reference.npz',**rows)
    model=dest/'model';model.mkdir()
    for name in ('base-weights.pt','preprocessing.npz'):shutil.copyfile(prepared/str(seed)/name,model/name)
    mechanism=checked_file(project/'runs/offline-accuracy/fuel-recipe-200k-20261009/dataset/mechanism.yaml',binding['mechanism_sha256'])
    shutil.copyfile(mechanism,model/'mechanism.yaml');save(model/'species.json',binding['species'])
    fit=dest/'fit-input';fit.mkdir();np.savez(fit/'training.npz',**rows);save(fit/'config.json',plan)
    populations=[('training',rows),*[(p,r) for p,r in population_rows(prepared,project/binding['transfer_root'],str(seed)) if p!='training']]
    for population,items in populations:
        inputs=dest/('input-'+population);inputs.mkdir()
        np.savez(inputs/'states.npz',states=items['states'],source_indices=items['source_indices']);save(inputs/'config.json',plan)


def execute(project,dest,output,seed,state,gpu):
    module='benchmarks.offline_accuracy.paper_baseline.reaction_attention.'
    elapsed=run_worker(project,dest/'fit-input',output,['--seed',str(seed)],state,gpu,dest/'model',worker_module=module+'worker')
    model=dest/'inference-model';shutil.copytree(dest/'model',model)
    for name in ('scaler.npz','graph.npz'):shutil.copyfile(output/name,model/name)
    for arm in read_json(dest/'fit-input/config.json')['arms']:
        (model/arm).mkdir()
        for name in ('weights.pt','config.json'):shutil.copyfile(output/arm/name,model/arm/name)
    for population in ('training','same_source_unseen','development'):
        elapsed+=run_worker(project,dest/('input-'+population),output/population,['--seed',str(seed)],state,gpu,model,worker_module=module+'infer')
    return elapsed
