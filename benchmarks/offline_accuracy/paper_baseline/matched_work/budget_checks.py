"""Read-only baseline and array identity gates for the longer fixed-data fit."""
import numpy as np

from .plan import configuration, extended_configuration


def check_baseline(result, recipe, seed, hashes):
    if (result["config"] != configuration(recipe, seed) or result["training_count"] != 200000
            or result["development_count"] != 1023 or result["independent_test_count"] != 0
            or result["updates_completed"] != 6000 or result["row_presentations"] != 60000000
            or result["hashes"] != hashes):
        raise ValueError("Require the same verified 200k, 6k-update baseline and data")


def check_pair(baseline_path, output_path, baseline, result):
    config = extended_configuration(result["config"]["recipe"], result["config"]["seed"])
    if result["config"] != config or result["updates_completed"] != 18000 or result["row_presentations"] != 180000000:
        raise ValueError("Extended fit has a different work budget")
    for key in ("hashes", "initial_weights_sha256", "preprocessing_array_sha256",
                "training_count", "development_count", "independent_test_count", "effective_batch_size"):
        if result[key] != baseline[key]:
            raise ValueError(f"Fixed-data pair differs: {key}")
    for name in ("training-indices.npy", "normalization-indices.npy"):
        np.testing.assert_array_equal(np.load(baseline_path / name, allow_pickle=False),
                                      np.load(output_path / name, allow_pickle=False))
    with np.load(baseline_path / "development-predictions.npz") as a, np.load(output_path / "development-predictions.npz") as b:
        np.testing.assert_array_equal(a["source_indices"], b["source_indices"])
