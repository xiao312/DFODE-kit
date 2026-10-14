"""Verify the nested-data and fixed-validation contract before comparing data sizes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision


def compare_arrays(small, large, label_small, label_large):
    if len(large["states"]) < len(small["states"]):
        raise ValueError("Second dataset must have at least as many states")
    count = len(small["states"])
    for name in small.files:
        np.testing.assert_array_equal(small[name], large[name][:count])
    np.testing.assert_array_equal(label_small["accepted"], label_large["accepted"][:count])
    mask = label_small["accepted"]
    difference = np.abs(label_small["delta"][mask] - label_large["delta"][:count][mask])
    budget = 1e-12 + 1e-6 * np.abs(small["states"][mask, 2:])
    maximum = float(np.max(difference / budget))
    if not np.isfinite(maximum) or maximum > .01:
        raise ValueError("Stored label differences exceed the reference agreement limit")
    return {"raw_states": count, "accepted_states": int(mask.sum()), "state_identity": True,
            "accepted_mask_identity": True, "label_difference_budget_max": maximum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("small", type=Path)
    parser.add_argument("large", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    _, _, first = load_dataset(args.small)
    _, _, second = load_dataset(args.large)
    if first["source_manifest"]["states_sha256"] != second["source_manifest"]["states_sha256"]:
        raise ValueError("Underlying flame source changed")
    if first["source_manifest"]["mechanism_sha256"] != second["source_manifest"]["mechanism_sha256"]:
        raise ValueError("Mechanism changed")
    ignored = {"train_count", "wall_seconds"}
    if {key: value for key, value in first["config"].items() if key not in ignored} != {key: value for key, value in second["config"].items() if key not in ignored}:
        raise ValueError("Configuration differs beyond data count and wall limit")
    result = {"status": "verified", "source": source_revision(), "splits": {},
              "small_manifest_sha256": sha256(args.small / "manifest.json"),
              "large_manifest_sha256": sha256(args.large / "manifest.json")}
    for split in ("train", "validation"):
        arrays = [np.load(root / split / "inputs.npz", allow_pickle=False) for root in (args.small, args.large)]
        labels = [np.load(root / split / "labels.npz", allow_pickle=False) for root in (args.small, args.large)]
        if split == "validation" and len(arrays[0]["states"]) != len(arrays[1]["states"]):
            raise ValueError("Validation count changed")
        result["splits"][split] = compare_arrays(*arrays, *labels)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
