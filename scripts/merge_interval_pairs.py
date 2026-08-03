#!/usr/bin/env python3
"""Merge interval-pairs-v1 HDF5 shards without loading them all in memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    inputs = sorted(args.input_dir.glob(args.pattern))
    if not inputs:
        raise FileNotFoundError(
            f"no inputs matched {args.input_dir / args.pattern}"
        )

    counts = []
    for path in inputs:
        with h5py.File(path, "r") as handle:
            counts.append(int(handle["pairs/current_states"].shape[0]))
    total = int(sum(counts))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(inputs[0], "r") as first, h5py.File(args.output, "w") as out:
        for key, value in first.attrs.items():
            out.attrs[key] = value
        out.attrs["n_pairs"] = total
        out.attrs["merged_shard_count"] = len(inputs)
        out.attrs["merged_inputs"] = json.dumps([str(path) for path in inputs])
        out.create_dataset(
            "species_names",
            data=first["species_names"][:],
            dtype=first["species_names"].dtype,
        )
        if "dt_bin_edges" in first:
            out.create_dataset("dt_bin_edges", data=first["dt_bin_edges"][:])
        if "dt_bin_counts" in first:
            bin_counts = np.zeros_like(first["dt_bin_counts"][:])
            for path in inputs:
                with h5py.File(path, "r") as handle:
                    bin_counts += handle["dt_bin_counts"][:]
            out.create_dataset("dt_bin_counts", data=bin_counts)

        output_pairs = out.create_group("pairs")
        for name, dataset in first["pairs"].items():
            shape = (total,) + dataset.shape[1:]
            output_pairs.create_dataset(
                name,
                shape=shape,
                dtype=dataset.dtype,
                chunks=True,
                compression="gzip",
            )

        offset = 0
        for path, count in zip(inputs, counts):
            with h5py.File(path, "r") as handle:
                for name in output_pairs:
                    output_pairs[name][offset : offset + count] = handle[
                        f"pairs/{name}"
                    ][:]
            offset += count

    with h5py.File(args.output, "r") as merged:
        max_delta = np.asarray(merged["pairs/max_abs_delta_y"], dtype=np.float64)
        temperature_delta = np.asarray(
            merged["pairs/temperature_delta"], dtype=np.float64
        )
    summary = {
        "inputs": [str(path) for path in inputs],
        "output": str(args.output),
        "shard_counts": counts,
        "n_pairs": total,
        "reactive_pair_fraction_1e-8": float(np.mean(max_delta > 1.0e-8)),
        "maximum_abs_delta_y": float(np.max(max_delta)),
        "maximum_abs_temperature_delta_k": float(
            np.max(np.abs(temperature_delta))
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
