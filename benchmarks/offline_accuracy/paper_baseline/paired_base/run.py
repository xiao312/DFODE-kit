"""Registered CPU-only arithmetic diagnostic; cached predictions are not a candidate."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, verify_evaluator_lock
from benchmarks.research_contract.replay import inputs, score
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..kinetic_features.review import recount, require
from .metrics import align_indices, means, agreement


def execute(project, plan, binding, output):
    prepared = project / binding["prepared_root"]
    selection = arrays(project / binding["transfer_root"] / "plan/selections.npz")
    full = []
    excluded = []
    for seed in plan["seeds"]:
        with np.load(prepared / str(seed) / "training.npz") as archive:
            row = {key: archive[key] for key in ("states", "delta", "base", "source_indices", "snapshot")}
        align_indices(row["source_indices"], row["source_indices"])
        full.append(row)
        excluded.extend(row["source_indices"][np.r_[np.arange(10000), selection[f"unseen_{seed}"]]])
    common = np.intersect1d(full[0]["source_indices"], full[1]["source_indices"])
    allowed = np.setdiff1d(common, np.unique(excluded))
    require(len(allowed) >= plan["rows"], "Insufficient common training support")
    ids = np.random.default_rng(plan["selection_seed"]).permutation(allowed)[:plan["rows"]]
    rows = [{key: value[align_indices(row["source_indices"], ids)] for key, value in row.items()} for row in full]
    del full
    for key in ("states", "delta", "source_indices", "snapshot"):
        np.testing.assert_array_equal(rows[0][key], rows[1][key])
    active = np.array([name != "AR" for name in binding["species"]])
    first, second = rows[0]["base"], rows[1]["base"]
    np.testing.assert_array_equal(first[:, ~active], second[:, ~active])
    candidates = means(first, second)
    for prediction in candidates.values():
        prediction[:, ~active] = first[:, ~active]
    output.mkdir(parents=True)
    save(output / "frozen-experiment.json", plan)
    save(output / "source.json", source_revision())
    np.savez(output / "predictions.npz", source_indices=ids, first=first, second=second, **candidates)
    replay = arrays(output / "predictions.npz")
    for name, value in means(replay["first"], replay["second"]).items():
        np.testing.assert_array_equal(value[:, active], replay[name][:, active])
    policy = read_json(project / "research/campaign.json")["screening"]
    scored, summary = [], []
    for seed, row in zip(plan["seeds"], rows):
        baseline = score(dict(prediction=row["base"], source_indices=ids), row, binding["species"], "paired-training")
        for name, prediction in candidates.items():
            result = score(dict(prediction=prediction, source_indices=ids), row, binding["species"], "paired-training")
            recount(prediction, row["delta"], row["base"], result, active, row["states"][:, 2:])
            group = result["groups"]["all"]
            scored.append(dict(comparator_seed=seed, arm=name, candidate=result, base=baseline))
            summary.append(dict(comparator_seed=seed, arm=name, acceptance=group["acceptance"],
                                base=group["base_acceptance"], repairs=group["repaired"], damage=group["damaged"],
                                damage_rate=group["damage_rate_among_base_successes"], p99=group["normalized_error"]["p99"],
                                negative=result["endpoint_diagnostics"]["negative_components"],
                                physical_gate=comparison(result, baseline, policy)))
    d = rows[0]["delta"][:, active]
    a, b = first[:, active], second[:, active]
    nontrivial = abs(d) > 1e-15 / .9
    masks = dict(all=np.ones_like(d, dtype=bool), floor=~nontrivial,
                 weak=nontrivial & (abs(d) <= 1e-12), active=abs(d) > 1e-12)
    groups = {key: agreement(a, b, d, mask) for key, mask in masks.items()}
    species = {}
    for index, name in enumerate(np.asarray(binding["species"])[active]):
        species[name] = {key: agreement(a[:, index], b[:, index], d[:, index], mask[:, index]) for key, mask in masks.items()}
    save(output / "scores.json", dict(records=scored))
    verify_evaluator_lock(project)
    save(output / "review.json", dict(status="verified_cached_diagnostic", summary=summary, groups=groups, per_species=species,
         namespace=binding["dataset_manifest_sha256"] + ":training", allowed_rows=len(allowed), excluded_union_count=len(np.unique(excluded)),
         predictions_sha256=digest(output / "predictions.npz"), scores_sha256=digest(output / "scores.json"),
         candidate=False, independent_replicates=1, inference_latency=None, confirmation_opened=False,
         limitations="One pair, both bases saw pool. Cached arithmetic, not fresh isolated inference. Oracle union uses labels for description only."))
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    plan = read_json(project / "research/experiments/EXP-20261013-paired-base-errors.json")
    output = project / plan["output"]
    if output.exists():
        raise ValueError("Existing diagnostic; do not overwrite")
    verify_evaluator_lock(project)
    checked_file(project / plan["parent_run"] / "review.json", plan["review_sha256"])
    binding = read_json(project / "research/bindings.json")
    inputs(project, binding)
    if source_revision()["dirty"]:
        raise ValueError("Commit the diagnostic before execution")
    print("PRECHECKED", flush=True)
    if args.execute:
        execute(project, plan, binding, output)


if __name__ == "__main__":
    main()
