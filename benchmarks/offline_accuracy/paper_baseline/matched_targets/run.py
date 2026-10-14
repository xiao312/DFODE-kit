"""Read-only input gate or one bounded GPU target/loss fit."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.paired.fit import weight_hash
from benchmarks.offline_accuracy.refinement.run import save
from ..matched_work.run import inputs, checked_result
from ..matched_work.fit import preprocessing_hash
from ..run import evaluate_and_verify
from .plan import configuration, TARGETS, OBJECTIVES, SEEDS
from .coordinates import preprocessing
from .model import prediction, reload_model
from .fit import fit
from .checks import check_baseline, check_result, check_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "previous", "original", "base-root", "baseline", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    config = configuration(args.target, args.objective, args.seed)
    training, normalization, development, physics, audit, hashes = inputs(
        args.campaign, args.previous, args.original, args.base_root, "fuel-state", args.seed)
    old_path = args.baseline / f"{args.seed}-fuel-state-u18000"
    old = checked_result(old_path)
    check_baseline(old, args.seed, hashes)
    for name, rows in (("training-indices.npy", training), ("normalization-indices.npy", normalization)):
        np.testing.assert_array_equal(np.load(old_path / name), rows["source_indices"])
    if physics["interval"] != config["interval"] or len(physics["species_names"]) != 59:
        raise ValueError("Unexpected chemistry interface")
    result = dict(status="planned", config=config, hashes=hashes, training_count=len(training["states"]),
                  development_count=len(development["states"]), independent_test_count=0,
                  baseline_result_sha256=sha256(old_path / "result.json"))
    print(json.dumps(result), flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source.get("dirty"):
        raise ValueError("Commit scientific source before executing")
    runtime = configure()
    args.output.mkdir(parents=True)
    result.update(status="running", source=source)
    save(args.output / "environment.json", runtime)
    save(args.output / "result.json", result)
    try:
        model, prep, fitted = fit(training, development, normalization, physics["species_names"], config, args.output)
        result.update(fitted)
        result.update(evaluate_and_verify(model, prep, training, development, physics, audit, config,
            args.output, args.campaign / "dataset", normalization_training=normalization,
            preprocess_fn=preprocessing, predict_fn=prediction, reload_fn=reload_model))
        check_arrays(old_path, args.output, same_target=args.target == "state-boxcox")
        if old["initial_weights_sha256"] != result["initial_weights_sha256"]:
            raise ValueError("Initial model differs from 18k control")
        if args.target == "state-boxcox" and args.objective == "coordinate":
            old_model, _ = reload_model(old_path, old["config"])
            if weight_hash(old_model) != weight_hash(model):
                raise ValueError("State-coordinate model does not reproduce 18k control")
            for split in ("training", "development"):
                with np.load(old_path / f"{split}-predictions.npz") as a, np.load(args.output / f"{split}-predictions.npz") as b:
                    for key in a.files:
                        np.testing.assert_array_equal(a[key], b[key])
            result["exact_historical_control_parity"] = True
        with np.load(args.output / "preprocessing.npz") as saved:
            if preprocessing_hash(dict(saved)) != result["preprocessing_array_sha256"]:
                raise ValueError("Saved normalization differs")
        result["artifacts"] = {name:sha256(args.output / name) for name in (
            "environment.json", "weights.pt", "optimizer.pt", "preprocessing.npz", "training-indices.npy",
            "normalization-indices.npy", "training-predictions.npz", "development-predictions.npz", "progress.jsonl")}
        result["status"] = "complete"
        check_result(result)
        save(args.output / "result.json", result)
        save(args.output / "verification.json", dict(status="verified", result_sha256=sha256(args.output / "result.json"),
            exact_model_replay=True, independent_paired_counts=True, physical_checks=True, frozen_50k_preprocessing=True))
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
