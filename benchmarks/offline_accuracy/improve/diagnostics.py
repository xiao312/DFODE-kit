"""Post-fit failure counts from checked prediction files; no model selection."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.offline_accuracy.refinement.run import save
from .plan import NAMES


def state_profile(prediction, reference, active):
    ratio = np.abs(prediction[:, active]-reference[:, active])/(1e-15+.1*np.abs(reference[:, active]))
    failing = np.sum(ratio > 1, axis=1)
    return dict(states=len(reference), minimum_failing_species=int(failing.min()),
                median_failing_species=float(np.median(failing)), maximum_failing_species=int(failing.max()),
                mean_failing_species=float(failing.mean()),
                state_max_error_median=float(np.median(ratio.max(axis=1))),
                failure_histogram=[dict(failing_species=int(value), states=int((failing == value).sum()))
                                   for value in range(int(active.sum())+1)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, _ = load_dataset(args.dataset)
    active = np.array([name != "AR" for name in physics["species_names"]])
    rows, files = [], []
    for seed in (20261011, 20261012):
        for name in NAMES:
            directory = args.results / f"seed-{seed}" / name
            result = json.loads((directory / "result.json").read_text())
            if result["hashes"]["dataset_manifest"] != sha256(args.dataset / "manifest.json"):
                raise ValueError("Dataset changed")
            if result["status"] == "failed":
                continue
            check = json.loads((directory / "verification.json").read_text())
            if check["status"] != "verified" or check["result_sha256"] != sha256(directory / "result.json"):
                raise ValueError("Require verified predictions")
            for split, source_split in (("training", "train"), ("validation", "validation")):
                file = directory / f"{split}-predictions.npz"
                if sha256(file) != result["artifacts"][file.name]:
                    raise ValueError("Prediction artifact changed")
                with np.load(file, allow_pickle=False) as arrays:
                    indices = np.searchsorted(data[source_split]["source_indices"], arrays["source_indices"])
                    np.testing.assert_array_equal(data[source_split]["source_indices"][indices], arrays["source_indices"])
                    profile = state_profile(arrays["prediction"], data[source_split]["delta"][indices], active)
                rows.append(dict(name=name, seed=seed, split=split, **profile))
            files.append(dict(name=f"seed-{seed}/{name}/result.json", sha256=sha256(directory / "result.json")))
    save(args.output, dict(status="complete", rows=rows, sources=files, dataset_manifest_sha256=sha256(args.dataset / "manifest.json")))


if __name__ == "__main__":
    main()
