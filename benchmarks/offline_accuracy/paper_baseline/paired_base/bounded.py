"""Registered single-bound follow-up to the paired-base diagnostic."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, verify_evaluator_lock
from benchmarks.research_contract.replay import inputs, score
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require, recount
from .metrics import align_indices, bounded_mean


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(); project = args.project.resolve()
    plan = read_json(project / "research/experiments/EXP-20261013-bounded-base-mean.json")
    output = project / plan["output"]; parent = project / plan["parent_run"]
    if output.exists(): raise ValueError("Existing result; do not overwrite")
    verify_evaluator_lock(project)
    review = read_json(checked_file(parent / "review.json", plan["review_sha256"]))
    saved = arrays(checked_file(parent / "predictions.npz", review["predictions_sha256"]))
    binding = read_json(project / "research/bindings.json"); inputs(project, binding)
    if source_revision()["dirty"]: raise ValueError("Commit before execution")
    print("PRECHECKED", flush=True)
    if not args.execute: return
    output.mkdir(parents=True); save(output / "frozen-experiment.json", plan); save(output / "source.json", source_revision())
    active = np.array([s != "AR" for s in binding["species"]]); policy = read_json(project / "research/campaign.json")["screening"]
    records, summary, predictions = [], [], dict(source_indices=saved["source_indices"])
    for seed, key, other_key in ((20261011, "first", "second"), (20261012, "second", "first")):
        full = arrays(project / binding["prepared_root"] / str(seed) / "training.npz")
        ix = align_indices(full["source_indices"], saved["source_indices"])
        rows = {k: v[ix] for k, v in full.items()}; del full
        np.testing.assert_array_equal(rows["base"], saved[key])
        b, q, d = saved[key], saved[other_key], rows["delta"]
        candidate = bounded_mean(b, q, plan["fraction"])
        predictions[str(seed)] = candidate
        # Independent scalar expression and convex-segment property.
        for index in np.linspace(0, len(b)-1, 32, dtype=int):
            for j in range(b.shape[1]):
                limit = plan["fraction"] * (1e-15 + .1 * abs(float(b[index, j])))
                step = (float(q[index, j]) - float(b[index, j])) / 2
                expected = float(b[index, j]) + min(abs(step), limit) * np.sign(step)
                require(abs(expected-candidate[index, j]) <= 4*np.spacing(max(abs(expected), 1e-30)), "Scalar formula differs")
        require(np.all(candidate >= np.minimum(b, q)) and np.all(candidate <= np.maximum(b, q)), "Not a convex-segment update")
        np.testing.assert_array_equal(candidate[:, ~active], b[:, ~active])
        error = abs(b-d)/(1e-15+.1*abs(d)); limit = (1-plan["fraction"])/(1+.1*plan["fraction"])
        protected = error[:, active] < limit-1e-10
        passed = abs(candidate[:, active]-d[:, active]) <= 1e-15+.1*abs(d[:, active])
        require(np.all(passed[protected]), "Preservation bound violated")
        result = score(dict(prediction=candidate, source_indices=rows["source_indices"]), rows, binding["species"], "paired-training")
        base = score(dict(prediction=b, source_indices=rows["source_indices"]), rows, binding["species"], "paired-training")
        recount(candidate, d, b, result, active, rows["states"][:, 2:]); g = result["groups"]["all"]
        records.append(dict(anchor_seed=seed, candidate=result, base=base))
        summary.append(dict(anchor_seed=seed, acceptance=g["acceptance"], base=g["base_acceptance"], repairs=g["repaired"], damage=g["damaged"],
                            damage_rate=g["damage_rate_among_base_successes"], p99=g["normalized_error"]["p99"],
                            negative=result["endpoint_diagnostics"]["negative_components"], guaranteed_region_components=int(protected.sum()),
                            physical_gate=comparison(result, base, policy)))
    np.savez(output / "predictions.npz", **predictions)
    save(output / "scores.json", dict(records=records)); verify_evaluator_lock(project)
    save(output / "review.json", dict(status="verified_cached_diagnostic", summary=summary, candidate=False,
         parent_review_sha256=plan["review_sha256"], predictions_sha256=digest(output / "predictions.npz"), scores_sha256=digest(output / "scores.json"),
         inference_latency=None, independent_pairs=1, confirmation_opened=False))
    print(summary, flush=True)


if __name__ == "__main__": main()
