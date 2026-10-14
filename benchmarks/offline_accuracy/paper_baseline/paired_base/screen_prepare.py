"""Trusted full-population staging; labels stay outside all worker mounts."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import read_json, checked_file, arrays
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save
from ..hidden_features.campaign import run_worker


def stage(project, dest, seed, plan):
    if plan.get("prior_screen_run"):
        checked_file(project/plan["prior_screen_run"]/"review.json",plan["prior_screen_review_sha256"])
    review = read_json(checked_file(project / plan["diagnostic_run"] / "review.json", plan["diagnostic_review_sha256"]))
    if not all(item["physical_gate"]["passed"] for item in review["summary"]):
        raise ValueError("Diagnostic physical gate did not pass")
    binding = read_json(project / "research/bindings.json")
    prepared, transfer = project / binding["prepared_root"], project / binding["transfer_root"]
    for other_seed in plan["seeds"]:
        model = dest / "model" / str(other_seed); model.mkdir(parents=True)
        for name in ("base-weights.pt", "preprocessing.npz"):
            shutil.copyfile(prepared / str(other_seed) / name, model / name)
    for population, rows in population_rows(prepared, transfer, str(seed)):
        if population not in plan["populations"]: continue
        if len(rows["states"]) != plan["counts"][population]: raise ValueError("Population size differs")
        np.savez(dest / (population + "-reference.npz"), **rows)
        inputs = dest / ("input-" + population); inputs.mkdir()
        np.savez(inputs / "states.npz", states=rows["states"], source_indices=rows["source_indices"])
        save(inputs / "config.json", plan)
    save(dest / "config.json", plan)


def execute(project, dest, output, seed, state, gpu):
    output.mkdir(); plan = read_json(dest / "config.json"); elapsed = 0.; records = []
    for population in plan["populations"]:
        elapsed += run_worker(project, dest / ("input-" + population), output / population,
                              ["--seed", str(seed)], state, gpu, dest / "model",
                              worker_module="benchmarks.offline_accuracy.paper_baseline.paired_base.screen_worker")
        records.append(dict(population=population, **read_json(output / population / "result.json")))
    save(output / "result.json", dict(status="complete", records=records))
    return elapsed
