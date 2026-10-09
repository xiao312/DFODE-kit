"""Discover or execute a source-grounded reduced-data Fuel recipe."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.refinement.run import inputs, save
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.metrics import summarize as increment_summary
from benchmarks.offline_accuracy.paired.metrics import summarize
from benchmarks.offline_accuracy.paired.verify import check_summary
from benchmarks.offline_accuracy.paired.runtime import configure
from .plan import configuration
from .coordinates import preprocessing
from .fit import fit, prediction, reload_model


def evaluate_and_verify(model, prep, training, validation, physics, audit, config, destination, dataset):
    restored, saved_prep = reload_model(destination, config)
    rebuilt = preprocessing(training, physics["species_names"], config)
    for key in rebuilt:
        np.testing.assert_array_equal(saved_prep[key], rebuilt[key])
    np.testing.assert_array_equal(np.load(destination / "training-indices.npy"), training["source_indices"])
    result = {}
    for name, rows in (("training", training), ("development", validation)):
        predicted, corrected = prediction(model, prep, rows["states"], config)
        replay, replay_corrected = prediction(restored, saved_prep, rows["states"], config)
        np.testing.assert_array_equal(predicted, replay)
        np.testing.assert_array_equal(corrected, replay_corrected)
        result[name] = summarize(predicted, rows["delta"], rows["states"][:, 2:], physics["species_names"])
        check_summary(predicted, rows["delta"], rows["states"][:, 2:], physics["species_names"], result[name])
        result[name+"_physical"] = physical_scores(predicted, rows["delta"], rows["states"], corrected, **physics)
        assert_scores(recompute(rows["states"], predicted, rows["delta"], dataset / "mechanism.yaml", physics["interval"]),
                      result[name+"_physical"])
        result[name+"_sspi"] = increment_summary(predicted, rows["delta"], physics["species_names"])["sspi"]
        np.savez_compressed(destination / f"{name}-predictions.npz", prediction=predicted,
                            correction=corrected, source_indices=rows["source_indices"])
    indices, reference, uncertainty = audit_subset(validation, audit)
    result["audited_subset"] = summarize(predicted[indices], reference, validation["states"][indices, 2:],
                                         physics["species_names"], uncertainty)
    check_summary(predicted[indices], reference, validation["states"][indices, 2:], physics["species_names"],
                  result["audited_subset"], uncertainty)
    result["zero_baseline"] = summarize(np.zeros_like(validation["delta"]), validation["delta"],
                                        validation["states"][:, 2:], physics["species_names"])
    import torch
    timings = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction(model, prep, validation["states"], config)
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-started)
    result["inference"] = dict(states=len(validation["states"]), wall_seconds=timings,
                                median_wall_seconds=float(np.median(timings)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    for name in ("audit", "base", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--recipe", choices=("fuel-state", "fuel-power"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--training-count", type=int, choices=(10000, 50000), default=10000)
    parser.add_argument("--comparison-dataset", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = configuration(args.recipe, args.seed)
    if args.output.exists():
        parser.error("Output already exists; select a new directory")
    runtime = configure()
    if args.training_count == 10000:
        if args.comparison_dataset is not None:
            parser.error("Comparison dataset is only for the 50k expansion")
        training, validation, physics, audit, _, _, hashes = inputs(args.dataset, args.audit, args.base, args.seed)
    else:
        if args.comparison_dataset is None:
            parser.error("The 50k expansion requires --comparison-dataset")
        from .data import expanded_inputs
        training, validation, physics, audit, hashes = expanded_inputs(
            args.dataset, args.audit, args.comparison_dataset, args.base, args.seed, args.training_count)
    if physics["interval"] != config["interval"] or len(physics["species_names"]) != 59:
        raise ValueError("Require the declared chemistry interval and mechanism")
    plan = dict(config=config, hashes=hashes, training_count=len(training["states"]),
                development_count=len(validation["states"]), independent_test_count=0,
                scope="Reduced-data source-recipe replication; no full-paper reproduction")
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    result = dict(plan, status="running", source=source_revision())
    save(args.output / "environment.json", runtime)
    save(args.output / "result.json", result)
    try:
        model, prep, fitted = fit(training, validation, physics["species_names"], config, args.output)
        result.update(fitted)
        result.update(evaluate_and_verify(model, prep, training, validation, physics, audit, config, args.output, args.dataset))
        result["artifacts"] = {name: sha256(args.output / name) for name in (
            "environment.json", "weights.pt", "optimizer.pt", "preprocessing.npz", "training-indices.npy",
            "training-predictions.npz", "development-predictions.npz", "progress.jsonl")}
        result["status"] = "complete"
        save(args.output / "result.json", result)
        save(args.output / "verification.json", dict(status="verified", result_sha256=sha256(args.output / "result.json"),
             exact_model_replay=True, independent_paired_counts=True, training_only_preprocessing=True, physical_checks=True))
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise
    print(json.dumps(dict(status="verified", recipe=args.recipe)), flush=True)


if __name__ == "__main__":
    main()
