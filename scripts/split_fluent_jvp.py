#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _copy_dataset(
    source: h5py.Dataset,
    destination: h5py.File,
    name: str,
    indices: np.ndarray,
    pair_count: int,
) -> None:
    if source.ndim > 0 and source.shape[0] == pair_count:
        data = source[indices]
    else:
        data = source[()]
    parent = str(Path(name).parent)
    if parent != ".":
        destination.require_group(parent)
    options: dict[str, object] = {}
    if source.ndim > 0 and source.compression is not None:
        options["compression"] = source.compression
        options["compression_opts"] = source.compression_opts
        options["shuffle"] = source.shuffle
    copied = destination.create_dataset(name, data=data, dtype=source.dtype, **options)
    for key, value in source.attrs.items():
        copied.attrs[key] = value


def _write_split(
    source_path: Path,
    output_path: Path,
    indices: np.ndarray,
    *,
    role: str,
    seed: int,
    test_fraction: float,
    pair_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["n_pairs"] = int(indices.size)
        output.attrs["split_role"] = role
        output.attrs["split_seed"] = int(seed)
        output.attrs["split_test_fraction"] = float(test_fraction)
        output.attrs["split_source"] = str(source_path.resolve())
        output.attrs["split_index_sha256"] = hashlib.sha256(
            np.ascontiguousarray(indices, dtype=np.int64).tobytes()
        ).hexdigest()

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                group = output.require_group(name)
                for key, value in obj.attrs.items():
                    group.attrs[key] = value
            else:
                _copy_dataset(obj, output, name, indices, pair_count)

        source.visititems(visitor)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation-pool and untouched-test JVP files."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("test-fraction must be strictly between zero and one")

    with h5py.File(args.source, "r") as source:
        if "pairs" not in source or "x" not in source["pairs"]:
            raise ValueError("source must use the fluent-di-jvp-v1 pairs/x schema")
        pair_count = int(source["pairs/x"].shape[0])
        if pair_count < 2:
            raise ValueError("source must contain at least two JVP pairs")

    generator = np.random.default_rng(args.seed)
    permutation = generator.permutation(pair_count)
    test_count = max(1, min(pair_count - 1, int(round(pair_count * args.test_fraction))))
    test_indices = np.sort(permutation[:test_count])
    trainval_indices = np.sort(permutation[test_count:])
    if np.intersect1d(trainval_indices, test_indices).size:
        raise RuntimeError("JVP split indices overlap")

    trainval_path = args.output_dir / "trainval.h5"
    test_path = args.output_dir / "test.h5"
    _write_split(
        args.source,
        trainval_path,
        trainval_indices,
        role="train-validation-pool",
        seed=args.seed,
        test_fraction=args.test_fraction,
        pair_count=pair_count,
    )
    _write_split(
        args.source,
        test_path,
        test_indices,
        role="untouched-test",
        seed=args.seed,
        test_fraction=args.test_fraction,
        pair_count=pair_count,
    )
    summary = {
        "schema": "dfode.fluent_di_jvp_split.v1",
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "source_pairs": pair_count,
        "trainval_pairs": int(trainval_indices.size),
        "test_pairs": int(test_indices.size),
        "trainval": str(trainval_path.resolve()),
        "test": str(test_path.resolve()),
    }
    summary_path = args.output_dir / "split-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
