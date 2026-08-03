#!/usr/bin/env python3
"""Split a CFD cell snapshot into deterministic row shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if args.shards < 1:
        raise ValueError("--shards must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as source:
        row_count = int(source["cells/states"].shape[0])
        row_slices = np.array_split(np.arange(row_count), args.shards)
        output_paths = []
        shard_counts = []
        for shard_index, row_indices in enumerate(row_slices):
            output = args.output_dir / f"{args.prefix}-{shard_index:04d}.h5"
            output_paths.append(str(output))
            shard_counts.append(int(row_indices.size))
            with h5py.File(output, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                target.attrs["sample_count"] = row_indices.size
                target.attrs["shard_index"] = shard_index
                target.attrs["shard_count"] = args.shards
                target.attrs["source_snapshot"] = str(args.source)
                cells = target.create_group("cells")
                for name, dataset in source["cells"].items():
                    if dataset.ndim < 1 or dataset.shape[0] != row_count:
                        continue
                    cells.create_dataset(
                        name,
                        data=dataset[row_indices],
                        compression="gzip",
                    )
                species = source["species_names"]
                target.create_dataset(
                    "species_names",
                    data=species[:],
                    dtype=species.dtype,
                )

    summary = {
        "source": str(args.source),
        "row_count": row_count,
        "shard_count": args.shards,
        "shard_counts": shard_counts,
        "outputs": output_paths,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
