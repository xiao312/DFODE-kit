"""Grow only the training pool; verify the old pool and development stay identical."""
import json
import numpy as np
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256


def nested_indices(old_count, new_count, selected_count, seed):
    if old_count < 10000 or not old_count <= new_count or not 10000 <= selected_count <= new_count:
        raise ValueError("Need the original 10k subset and enough accepted new rows")
    original = np.random.default_rng(seed).permutation(old_count)[:10000]
    rest = np.setdiff1d(np.arange(new_count), original)
    extra = np.random.default_rng(seed+1).permutation(rest)[:selected_count-10000]
    return np.concatenate([original, extra])


def expanded_inputs(dataset, audit_directory, comparison_dataset, base_directory, seed, count):
    if count != 50000:
        raise ValueError("Only the declared 50k expansion is supported")
    data, physics, manifest = load_dataset(dataset)
    old, _, original_manifest = load_dataset(comparison_dataset)
    new_config, old_config = dict(manifest["config"]), dict(original_manifest["config"])
    for key in ("train_count", "wall_seconds"):
        new_config.pop(key)
        old_config.pop(key)
    if new_config != old_config or manifest["source_manifest"] != original_manifest["source_manifest"]:
        raise ValueError("Expansion changed source, domain or chemistry configuration")
    for key, values in old["validation"].items():
        np.testing.assert_array_equal(data["validation"][key], values)
    for key, values in old["train"].items():
        np.testing.assert_array_equal(data["train"][key][:len(values)], values)
    audit = json.loads((audit_directory / "summary.json").read_text())
    digest = sha256(dataset / "manifest.json")
    if (audit["status"] != "complete" or not audit["reference_subset_pass"]
            or audit["dataset_manifest_sha256"] != digest):
        raise ValueError("Require a passing new reference subset audit")
    base = json.loads((base_directory / "summary.json").read_text())
    if base["dataset_manifest_sha256"] != sha256(comparison_dataset / "manifest.json") or base["plan"]["config"]["seed"] != seed:
        raise ValueError("Original comparison identity differs")
    indices = nested_indices(len(old["train"]["states"]), len(data["train"]["states"]), count, seed)
    training = {key: values[indices] for key, values in data["train"].items()}
    np.testing.assert_array_equal(np.load(base_directory / "n10000-float32-state-boxcox" / "training-indices.npy"),
                                  training["source_indices"][:10000])
    hashes = dict(dataset_manifest=digest, audit_summary=sha256(audit_directory / "summary.json"),
                  base_summary=sha256(base_directory / "summary.json"),
                  comparison_dataset_manifest=sha256(comparison_dataset / "manifest.json"))
    return training, data["validation"], physics, audit, hashes
