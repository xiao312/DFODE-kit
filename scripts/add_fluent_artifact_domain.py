#!/usr/bin/env python3
"""Add training-domain bounds to an exported Fluent artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as handle:
        states = np.asarray(handle["pairs/current_states"], dtype=np.float64)
        dt = np.asarray(handle["pairs/dt"], dtype=np.float64)
        species_names = [
            item.decode() if isinstance(item, bytes) else str(item)
            for item in handle["species_names"][:]
        ]

    inputs = np.concatenate((states, dt[:, None]), axis=1)
    lower_raw = np.min(inputs, axis=0)
    upper_raw = np.max(inputs, axis=0)
    lower = np.nextafter(
        lower_raw.astype(np.float32),
        np.float32(-np.inf),
    ).astype(np.float64)
    upper = np.nextafter(
        upper_raw.astype(np.float32),
        np.float32(np.inf),
    ).astype(np.float64)
    bounds = np.stack((lower, upper), axis=0)
    bounds.tofile(args.artifact_dir / "input_bounds.f64")

    summary = {
        "format": "dfode-input-domain-v1",
        "dataset": str(args.dataset),
        "sample_count": int(inputs.shape[0]),
        "input_width": int(inputs.shape[1]),
        "layout": ["temperature", "pressure", *species_names, "dt"],
        "minimum": lower.tolist(),
        "maximum": upper.tolist(),
    }
    (args.artifact_dir / "input_domain.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    runtime_path = args.artifact_dir / "runtime.cfg"
    runtime_text = runtime_path.read_text(encoding="utf-8")
    if "has_input_bounds=" not in runtime_text:
        runtime_path.write_text(
            runtime_text.rstrip() + "\nhas_input_bounds=1\n",
            encoding="utf-8",
        )

    manifest_path = args.artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_domain"] = {
        "format": "input_bounds.f64",
        "layout": "two contiguous float64 rows: minimum, maximum",
        "metadata": "input_domain.json",
        "sample_count": int(inputs.shape[0]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
