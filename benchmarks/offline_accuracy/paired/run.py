"""Read-only discovery or one bounded paired fit, followed by replay verification."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.offline_accuracy.refinement.run import inputs, save
from benchmarks.offline_accuracy.evaluate import audit_subset
from .fit import fit, prediction
from .metrics import summarize
from .plan import configuration


def evaluate(model, prep, training, validation, physics, audit, config, destination):
    result = {}
    predict = lambda states: prediction(model, prep, states, config)
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrections = predict(rows["states"])
        result[name] = summarize(predicted, rows["delta"], rows["states"][:, 2:], physics["species_names"])
        result[name+"_physical"] = physical_scores(predicted, rows["delta"], rows["states"], corrections, **physics)
        np.savez_compressed(destination / f"{name}-predictions.npz", prediction=predicted,
                            correction=corrections, source_indices=rows["source_indices"])
    indices, reference, uncertainty = audit_subset(validation, audit)
    result["audited_subset"] = summarize(predicted[indices], reference, validation["states"][indices, 2:],
                                         physics["species_names"], uncertainty)
    result["zero_baseline"] = summarize(np.zeros_like(validation["delta"]), validation["delta"],
                                        validation["states"][:, 2:], physics["species_names"])
    repeats = []
    for _ in range(5):
        wall, cpu = time.perf_counter(), time.process_time()
        predict(validation["states"])
        repeats.append(dict(wall_seconds=time.perf_counter()-wall, process_seconds=time.process_time()-cpu))
    result["inference"] = dict(states=len(validation["states"]), repeats=repeats,
        median_process_seconds=float(np.median([r["process_seconds"] for r in repeats])),
        median_wall_seconds=float(np.median([r["wall_seconds"] for r in repeats])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    for name in ("audit", "base", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = configuration(args.variant, args.seed)
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    if torch.__version__ != "2.5.1+cpu":
        raise ValueError("Use the existing pinned Torch image")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training, validation, physics, audit, _, _, hashes = inputs(args.dataset, args.audit, args.base, args.seed)
    if physics["interval"] != config["interval"]:
        raise ValueError("The declared interval must match the dataset")
    print(json.dumps(dict(config=config, hashes=hashes)), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    result = dict(status="running", config=config, hashes=hashes, source=source_revision(),
                  runtime=dict(torch=torch.__version__, threads=1, learning="float32", reconstruction="float64"))
    save(args.output / "result.json", result)
    try:
        model, prep, fitted = fit(training, physics["species_names"], config, args.output)
        result.update(fitted)
        result.update(evaluate(model, prep, training, validation, physics, audit, config, args.output))
        result["artifacts"] = {name: sha256(args.output / name) for name in (
            "weights.pt", "optimizer.pt", "preprocessing.npz", "training-indices.npy",
            "training-predictions.npz", "validation-predictions.npz")}
        result["status"] = "complete"
        save(args.output / "result.json", result)
        from .verify import verify
        verification = verify(args.dataset, args.audit, args.base, args.output)
        save(args.output / "verification.json", verification)
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise
    print(json.dumps(dict(status="verified", variant=args.variant)), flush=True)


if __name__ == "__main__":
    main()
