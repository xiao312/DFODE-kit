"""Exact saved-model replay and independent acceptance/physical checks."""
import json
import numpy as np
import torch
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.train import network
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.refinement.run import inputs
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.metrics import ABSOLUTE_FLOORS, RELATIVE_TOLERANCES, MAGNITUDE_EDGES
from .fit import preprocessing, prediction, reload_model, weight_hash
from .plan import configuration, POLICIES


def check_counts(error, budget, saved, uncertainty=None):
    passed = error <= budget
    expected = dict(components=error.size, states=error.shape[0],
        component_pass_count=np.count_nonzero(passed), state_pass_count=np.count_nonzero(np.all(passed, axis=1)),
        component_pass_fraction=np.mean(passed), state_pass_fraction=np.mean(np.all(passed, axis=1)))
    for key, value in expected.items():
        np.testing.assert_equal(saved[key], value)
    for key, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.)):
        np.testing.assert_equal(saved["normalized_quantiles"][key], np.quantile(error/budget, q))
    if uncertainty is not None:
        known = np.isfinite(uncertainty) & (uncertainty <= budget/10)
        accepted = known & (error+uncertainty <= budget)
        for key, value in dict(components=known.sum(), states=known.all(axis=1).sum(),
                               passed_components=accepted.sum(), passed_states=accepted.all(axis=1).sum()).items():
            np.testing.assert_equal(saved["qualification"][key], value)


def check_summary(predicted, reference, initial, species, saved, uncertainty=None):
    mask = np.array([s != "AR" for s in species])
    error = np.abs(predicted[:, mask]-reference[:, mask])
    truth, start = reference[:, mask], initial[:, mask]
    uncertainty = None if uncertainty is None else uncertainty[:, mask]
    if set(saved) != set(POLICIES):
        raise ValueError("Missing paired policy")
    for policy in POLICIES:
        magnitude = np.abs(truth) if policy == POLICIES[0] else np.abs(start+truth)
        pairs = [(a, r, "paired-grid") for a in ABSOLUTE_FLOORS for r in RELATIVE_TOLERANCES]
        if policy == POLICIES[1]:
            pairs.append((1e-12, 1e-6, "application-state-diagnostic"))
        rows = saved[policy]["grid"]
        if len(rows) != len(pairs):
            raise ValueError("Tolerance grid differs")
        for row, (a, r, role) in zip(rows, pairs):
            if (row["atol"], row["rtol"], row["role"]) != (a, r, role):
                raise ValueError("Tolerance identity differs")
            check_counts(error, a+r*magnitude, row, uncertainty)
        budget = 1e-15+.1*magnitude
        names = np.array(species)[mask]
        if len(saved[policy]["per_species"]) != len(names):
            raise ValueError("Species population differs")
        for j, (name, row) in enumerate(zip(names, saved[policy]["per_species"])):
            if name != row["species"]:
                raise ValueError("Species identity differs")
            check_counts(error[:, j:j+1], budget[:, j:j+1], row)
        bins = saved[policy]["magnitude_bins"]
        if len(bins) != len(MAGNITUDE_EDGES)-1:
            raise ValueError("Magnitude bins differ")
        for row, low, high in zip(bins, MAGNITUDE_EDGES[:-1], MAGNITUDE_EDGES[1:]):
            selected = (np.abs(truth) >= low) & (np.abs(truth) < high)
            assert row["lower"] == low and row["upper"] == (None if np.isinf(high) else high)
            assert row["components"] == selected.sum()
            assert row["component_pass_count"] == ((error <= budget) & selected).sum()


def verify(dataset, audit_directory, base_directory, directory):
    result = json.loads((directory / "result.json").read_text())
    config = result["config"]
    if result["status"] != "complete" or config != configuration(config["name"], config["seed"]):
        raise ValueError("Require a complete frozen configuration")
    if result["updates_completed"] != config["updates"]:
        raise ValueError("Incomplete fit")
    training, validation, physics, audit, _, _, hashes = inputs(dataset, audit_directory, base_directory, config["seed"])
    if result["hashes"] != hashes:
        raise ValueError("Input hashes differ")
    for name, digest in result["artifacts"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Artifact changed: {name}")
    np.testing.assert_array_equal(np.load(directory / "training-indices.npy"), training["source_indices"])
    model, prep = reload_model(directory, config)
    rebuilt = preprocessing(training, physics["species_names"], config)
    for key in rebuilt:
        np.testing.assert_array_equal(prep[key], rebuilt[key])
    initial = network(training["states"].shape[1], training["delta"].shape[1], config["widths"],
                      config["seed"], torch.float32, "gelu")
    if weight_hash(initial) != result["initial_weights_sha256"]:
        raise ValueError("Initialization differs")
    assert result["parameter_count"] == sum(p.numel() for p in model.parameters())
    for name, rows in (("training", training), ("validation", validation)):
        predicted, corrected = prediction(model, prep, rows["states"], config)
        with np.load(directory / f"{name}-predictions.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["prediction"], predicted)
            np.testing.assert_array_equal(arrays["correction"], corrected)
            np.testing.assert_array_equal(arrays["source_indices"], rows["source_indices"])
        check_summary(predicted, rows["delta"], rows["states"][:, 2:], physics["species_names"], result[name])
        independent = recompute(rows["states"], predicted, rows["delta"], dataset / "mechanism.yaml", physics["interval"])
        assert_scores(independent, result[name+"_physical"])
    indices, reference, uncertainty = audit_subset(validation, audit)
    check_summary(predicted[indices], reference, validation["states"][indices, 2:], physics["species_names"],
                  result["audited_subset"], uncertainty)
    check_summary(np.zeros_like(validation["delta"]), validation["delta"], validation["states"][:, 2:],
                  physics["species_names"], result["zero_baseline"])
    assert result["training_process_seconds"] > 0 and result["training_wall_seconds"] > 0
    timing = result["inference"]
    assert timing["states"] == len(validation["states"]) and len(timing["repeats"]) == 5
    for key in ("process_seconds", "wall_seconds"):
        values = [r[key] for r in timing["repeats"]]
        assert np.isfinite(values).all() and min(values) > 0
        np.testing.assert_equal(timing["median_"+key], np.median(values))
    return dict(status="verified", result_sha256=sha256(directory / "result.json"), exact_model_replay=True,
                independent_paired_counts=True, physical_checks=True, training_only_preprocessing=True)
