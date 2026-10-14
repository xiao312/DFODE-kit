"""Trusted staging and execution; inference receives no reference arrays."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file
from ..selective_correction.prepare import stage as select_rows
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker


def stage(project, dest, seed, plan):
    # This controller-only directory is never mounted as worker input.
    select_rows(project, dest, seed, plan)
    rows = arrays(dest/"training.npz")
    fit_input = dest/"fit-input"; fit_input.mkdir()
    np.savez(fit_input/"training.npz", **{k:v[:plan["fit_rows"]] for k,v in rows.items()})
    save(fit_input/"config.json", plan)
    infer_input = dest/"inference-input"; infer_input.mkdir()
    np.savez(infer_input/"states.npz", states=rows["states"], source_indices=rows["source_indices"])
    save(infer_input/"config.json", plan)
    binding = read_json(project/"research/bindings.json")
    source = checked_file(project/"runs/offline-accuracy/fuel-recipe-200k-20261009/dataset/mechanism.yaml", binding["mechanism_sha256"])
    shutil.copyfile(source, dest/"model/mechanism.yaml")
    save(dest/"model/species.json", binding["species"])


def execute(project, dest, output, seed, state, gpu):
    elapsed = run_worker(project, dest/"fit-input", output, ["--seed",str(seed)], state, gpu,
        dest/"model", worker_module="benchmarks.offline_accuracy.paper_baseline.kinetic_features.worker")
    plan = read_json(dest/"config.json")
    # Build a strict model-only mount, never mount fitting outputs wholesale.
    model = dest/"inference-model"; model.mkdir()
    for name in ("base-weights.pt","preprocessing.npz","mechanism.yaml","species.json"):
        shutil.copyfile(dest/"model"/name, model/name)
    shutil.copyfile(output/"scaler.npz", model/"scaler.npz")
    for arm in plan["arms"]:
        (model/arm).mkdir()
        for name in ("weights.pt","config.json"):
            shutil.copyfile(output/arm/name, model/arm/name)
    return elapsed + run_worker(project, dest/"inference-input", output/"inference",
        ["--seed",str(seed)], state, gpu, model,
        worker_module="benchmarks.offline_accuracy.paper_baseline.kinetic_features.infer")
