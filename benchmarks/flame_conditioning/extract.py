"""Read paired original line samples without changing a study case.

The file names define column order. Row identity is a sampled coordinate, not
necessarily an original finite-volume cell. Explicit closure normalization is
recorded because text samples have limited precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cantera as ct
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision():
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    return {"commit": commit, "dirty": bool(dirty)}


def normalize_sampled_states(states, maximum_closure_error=1e-4):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 3 or not np.isfinite(states).all():
        raise ValueError("Expected finite rows [T, p, Y...]")
    if np.any(states[:, :2] <= 0) or np.any(states[:, 2:] < 0):
        raise ValueError("Source has non-positive T/p or negative Y; do not silently clip")
    totals = states[:, 2:].sum(axis=1)
    if np.any(np.abs(totals - 1) > maximum_closure_error):
        raise ValueError("Source mass closure exceeds the declared text-rounding limit")
    corrected = states.copy()
    corrected[:, 2:] /= totals[:, None]
    return corrected, {
        "raw_mass_sum_error": totals - 1,
        "normalization_max_absolute_change": np.max(np.abs(corrected[:, 2:] - states[:, 2:]), axis=1),
    }


def read_snapshot(case, time_name, species_names):
    values, fields, files = [], [], []
    coordinates = None
    for suffix in ("A", "B"):
        directory = Path(case) / "postProcessing" / f"lineSample{suffix}" / time_name
        candidates = list(directory.glob(f"line{suffix}_*.xy"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one paired line sample in {directory}; found {len(candidates)}")
        path = candidates[0]
        names = path.stem.removeprefix(f"line{suffix}_").split("_")
        array = np.loadtxt(path, ndmin=2)
        if array.shape[1] != 1 + len(names) or not np.isfinite(array).all():
            raise ValueError(f"Invalid sample shape or non-finite values: {path.name}")
        if coordinates is None:
            coordinates = array[:, 0]
        elif not np.array_equal(coordinates, array[:, 0]):
            raise ValueError("Paired line samples have different coordinates or row order")
        values.append(array[:, 1:])
        fields.extend(names)
        files.append({"relative_path": str(path.relative_to(case)), "sha256": sha256(path)})
    if fields != ["T", "p"] + list(species_names):
        raise ValueError("Line sample columns do not match mechanism species order")
    if len(coordinates) < 2 or np.any(np.diff(coordinates) <= 0):
        raise ValueError("Sample coordinates must be strictly increasing")
    raw = np.column_stack(values)
    corrected, normalization = normalize_sampled_states(raw)
    return raw, corrected, coordinates, normalization, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--times", nargs="+", default=["0.0001", "0.0005", "0.001", "0.0015", "0.002", "0.0025"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.case.is_dir() or not args.mechanism.is_file():
        parser.error("Case directory and mechanism file must exist")
    if args.output.exists():
        parser.error("Output already exists; choose a new run directory")
    if len(set(args.times)) != len(args.times) or any(not np.isfinite(float(t)) or float(t) < 0 for t in args.times):
        parser.error("Snapshot times must be distinct finite nonnegative values")
    gas = ct.Solution(str(args.mechanism))
    snapshots = [read_snapshot(args.case, value, gas.species_names) for value in args.times]
    rows = sum(len(snapshot[0]) for snapshot in snapshots)
    plan = {"rows": rows, "times": args.times, "species_count": gas.n_species,
            "reaction_count": gas.n_reactions, "mechanism_sha256": sha256(args.mechanism),
            "maximum_raw_closure_error": max(float(np.abs(s[3]["raw_mass_sum_error"]).max()) for s in snapshots)}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.mechanism, args.output / "mechanism.yaml")
    np.savez_compressed(
        args.output / "source-states.npz",
        raw=np.concatenate([s[0] for s in snapshots]),
        states=np.concatenate([s[1] for s in snapshots]),
        coordinate=np.concatenate([s[2] for s in snapshots]),
        snapshot=np.concatenate([np.full(len(s[0]), name) for name, s in zip(args.times, snapshots, strict=True)]),
        sample_row=np.concatenate([np.arange(len(s[0])) for s in snapshots]),
        raw_mass_sum_error=np.concatenate([s[3]["raw_mass_sum_error"] for s in snapshots]),
        normalization_max_absolute_change=np.concatenate([s[3]["normalization_max_absolute_change"] for s in snapshots]),
    )
    manifest = {
        **plan, "status": "extracted", "source": source_revision(), "cantera": ct.__version__,
        "numpy": np.__version__, "source_case": str(args.case.resolve()),
        "species_names": gas.species_names, "source_files": [f for s in snapshots for f in s[4]],
        "source_precision": "Original ASCII samples: approximately six significant digits",
        "normalization": "Divide each nonnegative Y row by its raw sum; no clipping",
        "independence": "All snapshots belong to one flame realization; not independent flames",
        "mechanism_provenance": "User study asset; redistribution license not established",
        "states_sha256": sha256(args.output / "source-states.npz"),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
