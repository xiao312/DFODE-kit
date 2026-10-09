"""Gate a fixed-work fit on completed, immutable nested source experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.refinement.run import save
from ..data import expanded_inputs
from ..plan import configuration as source_configuration
from ..run import evaluate_and_verify
from .fit import fit, preprocessing_hash
from .plan import configuration


def checked_result(path):
    result = json.loads((path / "result.json").read_text())
    check = json.loads((path / "verification.json").read_text())
    if (result["status"] != "complete" or check["status"] != "verified" or
            check["result_sha256"] != sha256(path / "result.json")):
        raise ValueError("Source result is incomplete or changed")
    for name, digest in result["artifacts"].items():
        if Path(name).name != name or sha256(path / name) != digest:
            raise ValueError("Source artifact checksum mismatch")
    return result


def inputs(campaign, previous, original, base_root, recipe, seed):
    for root in (campaign, previous):
        if json.loads((root / "campaign-status.json").read_text())["status"] != "complete":
            raise ValueError("Both source campaigns must be complete")
    full_path, small_path = [root / f"seed-{seed}" / recipe for root in (campaign, previous)]
    full, small = checked_result(full_path), checked_result(small_path)
    expected = source_configuration(recipe, seed)
    if (full["config"] != expected or small["config"] != expected or
            full["training_count"] != 200000 or small["training_count"] != 50000 or
            full["hashes"]["nested_50k_result"] != sha256(small_path / "result.json")):
        raise ValueError("Source recipe/seed/nesting differs")
    training, development, physics, audit, hashes = expanded_inputs(
        campaign / "dataset", campaign / "audit", original,
        base_root / f"seed-{seed}" / "training", seed, 200000, small_path)
    if (full["hashes"]["dataset_manifest"] != hashes["dataset_manifest"] or
            full["hashes"]["audit_summary"] != hashes["audit_summary"]):
        raise ValueError("Completed 200k run does not match current dataset/audit")
    np.testing.assert_array_equal(training["source_indices"], np.load(full_path / "training-indices.npy", allow_pickle=False))
    normalization = {key: value[:50000] for key, value in training.items()}
    np.testing.assert_array_equal(normalization["source_indices"], np.load(small_path / "training-indices.npy", allow_pickle=False))
    hashes.update(source_200k_result=sha256(full_path / "result.json"), source_50k_result=sha256(small_path / "result.json"))
    return training, normalization, development, physics, audit, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "previous", "original", "base-root", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--recipe", choices=("fuel-state", "fuel-power"), required=True)
    parser.add_argument("--seed", type=int, choices=(20261011, 20261012), required=True)
    parser.add_argument("--training-count", type=int, choices=(50000, 200000), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run")
    config = configuration(args.recipe, args.seed)
    full, normalization, development, physics, audit, hashes = inputs(
        args.campaign, args.previous, args.original, args.base_root, args.recipe, args.seed)
    training = full if args.training_count == 200000 else normalization
    if physics["interval"] != 1e-6 or len(physics["species_names"]) != 59:
        raise ValueError("Unexpected chemistry interface")
    result = dict(status="planned", config=config, hashes=hashes, training_count=args.training_count,
                  development_count=len(development["states"]), independent_test_count=0,
                  scope="Fixed work and fixed 50k normalization; not original epoch schedule")
    print(json.dumps(result), flush=True)
    if not args.execute:
        return
    runtime = configure()
    args.output.mkdir(parents=True)
    result.update(status="running", source=source_revision())
    save(args.output / "environment.json", runtime)
    save(args.output / "result.json", result)
    try:
        model, prep, fitted = fit(training, development, normalization, physics["species_names"], config, args.output)
        result.update(fitted)
        result.update(evaluate_and_verify(model, prep, training, development, physics, audit, config,
                      args.output, args.campaign / "dataset", normalization_training=normalization))
        with np.load(args.output / "preprocessing.npz", allow_pickle=False) as saved:
            if preprocessing_hash(dict(saved)) != result["preprocessing_array_sha256"]:
                raise ValueError("Saved normalization differs")
        result["artifacts"] = {name:sha256(args.output / name) for name in (
            "environment.json", "weights.pt", "optimizer.pt", "preprocessing.npz", "training-indices.npy",
            "normalization-indices.npy", "training-predictions.npz", "development-predictions.npz", "progress.jsonl")}
        result["status"] = "complete"
        save(args.output / "result.json", result)
        save(args.output / "verification.json", dict(status="verified", result_sha256=sha256(args.output / "result.json"),
            exact_model_replay=True, independent_paired_counts=True, physical_checks=True, frozen_50k_preprocessing=True))
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    main()
