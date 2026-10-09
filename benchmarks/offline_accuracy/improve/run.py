"""Validate, run, save and independently check one acceptance adaptation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import scipy
import torch

from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.metrics import summarize
from benchmarks.offline_accuracy.verify_evaluation import check_summary
from benchmarks.offline_accuracy.refinement.fit import reload_model, prediction, weight_hash
from benchmarks.offline_accuracy.refinement.run import inputs, save
from . import arrhenius, local, neural
from .coordinates import transitions
from .plan import configuration, NAMES


def load_inputs(args):
    training, validation, physics, audit, _, _, hashes = inputs(args.dataset, args.audit, args.base, args.seed)
    result = json.loads((args.refined / "result.json").read_text())
    check = json.loads((args.refined / "verification-final.json").read_text())
    if (result["status"] != "complete" or result["config"]["name"] != "long-state-boxcox"
            or result["config"]["seed"] != args.seed or result["hashes"] != hashes
            or check["status"] != "verified" or check["result_sha256"] != sha256(args.refined / "result.json")):
        raise ValueError("Require matching verified long conventional base")
    for name, digest in result["artifacts"].items():
        if sha256(args.refined / name) != digest:
            raise ValueError(f"Frozen long-base artifact changed: {name}")
    np.testing.assert_array_equal(np.load(args.refined / "training-indices.npy"), training["source_indices"])
    model, preprocessing = reload_model(args.refined, result["config"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hashes.update(long_base_result=sha256(args.refined / "result.json"),
                  long_base_verification=sha256(args.refined / "verification-final.json"))
    return training, validation, physics, audit, model, preprocessing, result, hashes


def evaluate(predict, base, training, validation, physics, audit, destination):
    result = {}
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrections = predict(rows["states"])
        original, _ = base(rows["states"])
        result[name] = summarize(predicted, rows["delta"], physics["species_names"])
        result[f"{name}_physical"] = physical_scores(predicted, rows["delta"], rows["states"], corrections, **physics)
        result[f"{name}_transitions"] = transitions(original, predicted, rows["delta"], physics["species_names"])
        np.savez_compressed(destination / f"{name}-predictions.npz", prediction=predicted,
                            correction=corrections, source_indices=rows["source_indices"])
    selected, reference, uncertainty = audit_subset(validation, audit)
    result["audited_subset"] = summarize(predicted[selected], reference, physics["species_names"], uncertainty)
    result["zero_baseline"] = summarize(np.zeros_like(validation["delta"]), validation["delta"], physics["species_names"])
    durations = []
    for _ in range(5):
        wall, cpu = time.perf_counter(), time.process_time()
        predict(validation["states"])
        durations.append(dict(wall_seconds=time.perf_counter()-wall, process_seconds=time.process_time()-cpu))
    result["inference"] = dict(states=len(validation["states"]), repeats=durations,
        median_process_seconds=float(np.median([r["process_seconds"] for r in durations])),
        median_wall_seconds=float(np.median([r["wall_seconds"] for r in durations])))
    return result


def verify(args, training, validation, physics, audit, model, p, hashes):
    directory = args.output
    result = json.loads((directory / "result.json").read_text())
    config = configuration(args.variant, args.seed)
    if result["status"] != "complete" or result["config"] != config or result["hashes"] != hashes:
        raise ValueError("Incomplete or inconsistent result")
    for name, digest in result["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Changed saved artifact: {name}")
    np.testing.assert_array_equal(np.load(directory / "training-indices.npy"), training["source_indices"])
    new_inputs = config["name"].startswith("arrhenius-")
    predictor = (arrhenius.reload(directory, config) if new_inputs else
                 local.reload(directory, config) if config["local"] else neural.reload(directory, config, model))
    if new_inputs:
        x, target, expected = arrhenius.prepare(training, physics["molecular_weights"], config, p["active"])
        for key, value in expected.items():
            np.testing.assert_array_equal(predictor.preprocessing[key], value)
        if config["local"]:
            _, singular, vh = np.linalg.svd(x, full_matrices=False)
            basis = vh[singular > singular[0]*1e-10].T
            np.testing.assert_array_equal(predictor.preprocessing["basis"], basis)
            np.testing.assert_array_equal(predictor.preprocessing["points"], x @ basis)
            np.testing.assert_array_equal(predictor.preprocessing["targets"], target)
    elif not config["local"]:
        for key, value in p.items():
            np.testing.assert_array_equal(predictor.preprocessing[key], value)
    else:
        # Rebuild the training-only table, including rank reduction, in memory.
        a = predictor.arrays
        from benchmarks.flame_conditioning.coordinates import input_features, state_change
        x = (input_features(training["states"])-p["x_offset"])/p["x_scale"]
        _, singular, vh = np.linalg.svd(x, full_matrices=False)
        np.testing.assert_array_equal(a["basis"], vh[singular > singular[0]*1e-10].T)
        np.testing.assert_array_equal(a["points"], x @ a["basis"])
        target = (state_change(training["states"][:, 2:], training["delta"]) if config["name"] == "local-state"
                  else np.arcsinh(training["delta"]/1e-14))
        scale = np.maximum(np.sqrt(np.mean(target**2, axis=0)), 1e-30)
        np.testing.assert_array_equal(a["y_scale"], scale)
        np.testing.assert_array_equal(a["targets"], target/scale)
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrected = predictor(rows["states"])
        with np.load(directory / f"{name}-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["prediction"], predicted)
            np.testing.assert_array_equal(saved["correction"], corrected)
            np.testing.assert_array_equal(saved["source_indices"], rows["source_indices"])
        check_summary(predicted, rows["delta"], result[name], physics["species_names"])
        actual = recompute(rows["states"], predicted, rows["delta"], args.dataset / "mechanism.yaml", physics["interval"])
        assert_scores(actual, result[f"{name}_physical"])
        original, _ = prediction(model, p, rows["states"], "state-boxcox")
        if transitions(original, predicted, rows["delta"], physics["species_names"]) != result[f"{name}_transitions"]:
            raise ValueError("Transition counts differ")
    selected, reference, uncertainty = audit_subset(validation, audit)
    check_summary(predicted[selected], reference, result["audited_subset"], physics["species_names"], uncertainty)
    # Re-read frozen files and original data identities after the run.
    if load_inputs(args)[-1] != hashes:
        raise ValueError("Frozen inputs changed")
    return dict(status="verified", result_sha256=sha256(directory / "result.json"),
                model_replay="exact", independent_counts=True, physical_checks=True,
                training_only_preprocessing=True, frozen_inputs_unchanged=True, transitions_checked=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    for name in ("audit", "base", "refined", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=NAMES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = configuration(args.variant, args.seed)
    if args.output.exists():
        parser.error("Output must be a new directory")
    if torch.__version__ != "2.5.1+cpu":
        raise ValueError("Use the existing pinned research image")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training, validation, physics, audit, model, p, original, hashes = load_inputs(args)
    print(json.dumps(dict(config=config, hashes=hashes)), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    result = dict(status="running", config=config, hashes=hashes, source=source_revision(),
                  training_count=len(training["states"]),
                  runtime=dict(torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__, threads=1))
    frozen_hash = weight_hash(model)
    save(args.output / "result.json", result)
    try:
        new_inputs = config["name"].startswith("arrhenius-")
        if new_inputs:
            predictor, fitted = arrhenius.fit(training, validation, physics, config, p["active"], args.output)
        elif config["local"]:
            predictor, fitted = local.fit(training, config, p, args.output)
        else:
            predictor, fitted = neural.fit(training, validation, config, model, p, args.output)
        result.update(fitted)
        base = lambda states: prediction(model, p, states, "state-boxcox")
        result.update(evaluate(predictor, base, training, validation, physics, audit, args.output))
        result["base_training_process_seconds"] = original["training_process_seconds"] if not (config["local"] or new_inputs) else 0.
        result["total_training_process_seconds"] = result["fit_process_seconds"] + result["base_training_process_seconds"]
        result["total_parameter_count"] = (None if config["local"] else result["parameter_count"] +
            (sum(v.numel() for v in model.parameters()) if config["correction"] else 0))
        np.save(args.output / "training-indices.npy", training["source_indices"])
        if weight_hash(model) != frozen_hash:
            raise ValueError("Frozen model parameters changed")
        result["artifacts"] = {path.name: sha256(path) for path in args.output.iterdir() if path.suffix in (".npz", ".npy", ".pt")}
        result["status"] = "complete"
        save(args.output / "result.json", result)
        checked = verify(args, training, validation, physics, audit, model, p, hashes)
        save(args.output / "verification.json", checked)
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "result.json", result)
        raise
    primary = next(row for row in result["validation"]["tolerances"] if row["atol"] == 1e-15 and row["rtol"] == .1)
    print(json.dumps(dict(status="verified", name=args.variant, primary=primary)), flush=True)


if __name__ == "__main__":
    main()
