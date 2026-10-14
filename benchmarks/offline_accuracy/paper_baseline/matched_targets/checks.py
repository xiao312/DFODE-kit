"""Fail-closed paired identities, baseline parity and complete-matrix checks."""
import numpy as np
from .plan import configuration, TARGETS, OBJECTIVES, SEEDS
from ..matched_work.plan import extended_configuration


def check_baseline(result, seed, hashes):
    if (result["config"] != extended_configuration("fuel-state", seed)
            or result["hashes"] != hashes or result["training_count"] != 200000
            or result["development_count"] != 1023 or result["independent_test_count"] != 0
            or result["updates_completed"] != 18000 or result["row_presentations"] != 180000000):
        raise ValueError("Require the verified same-data 18k state baseline")


def check_result(result):
    config = result["config"]
    if (config != configuration(config["target"], config["objective"], config["seed"])
            or result["status"] != "complete" or result["training_count"] != 200000
            or result["development_count"] != 1023 or result["independent_test_count"] != 0
            or result["updates_completed"] != 18000 or result["row_presentations"] != 180000000
            or result["effective_batch_size"] != 10000 or not result["warmup_weights_sha256"]):
        raise ValueError("Incomplete or altered matched-target result")
    history = result["history"]
    if [r["updates"] for r in history] != list(range(1000, 18001, 1000)):
        raise ValueError("Incomplete diagnostic history")
    for row in history:
        expected = "coordinate" if row["updates"] <= 12000 else config["objective"]
        if row["objective"] != expected or row["row_presentations"] != row["updates"]*10000:
            raise ValueError("Diagnostic work or objective differs")


def check_pair(first, second, same_target):
    for key in ("hashes", "source", "initial_weights_sha256", "baseline_result_sha256"):
        if first[key] != second[key]:
            raise ValueError(f"Paired {key} differs")
    if same_target:
        for key in ("warmup_weights_sha256", "preprocessing_array_sha256"):
            if first[key] != second[key]:
                raise ValueError(f"Objective-paired {key} differs")
        for a, b in zip(first["history"][:12], second["history"][:12]):
            # Wall time can differ, but every scientific warmup diagnostic must match.
            for key in ("loss", "training", "development"):
                if a[key] != b[key]:
                    raise ValueError("Objective-paired warmup diagnostics differ")


def check_arrays(first_path, second_path, same_target):
    for filename in ("training-indices.npy", "normalization-indices.npy"):
        np.testing.assert_array_equal(np.load(first_path / filename), np.load(second_path / filename))
    with np.load(first_path / "preprocessing.npz") as a, np.load(second_path / "preprocessing.npz") as b:
        keys = a.files if same_target else ("x_offset", "x_scale", "x_constant", "active")
        for key in keys:
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(first_path / "development-predictions.npz") as a, np.load(second_path / "development-predictions.npz") as b:
        np.testing.assert_array_equal(a["source_indices"], b["source_indices"])


def matrix_names():
    return [f"{seed}-{target}-{objective}" for seed in SEEDS for target in TARGETS for objective in OBJECTIVES]
