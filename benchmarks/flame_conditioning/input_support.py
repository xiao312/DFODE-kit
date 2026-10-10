"""Post-score input coverage diagnosis; no fitting, inference, or scaler changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.coordinates import input_features, standardization
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision


def profile_population(train, observed, offset, scale, names, population):
    if not len(observed):
        raise ValueError("Cannot profile an empty population")
    encoded = input_features(observed)
    standardized = np.abs((encoded - offset) / scale)
    rows = []
    for i, name in enumerate(names):
        low, high = float(train[:, i].min()), float(train[:, i].max())
        rows.append({"feature": name, "population": population, "samples": len(observed),
                     "encoding": "identity" if i < 2 else "mass_fraction**0.1",
                     "trainingMin": low, "trainingMax": high,
                     "observedMin": float(observed[:, i].min()), "observedMax": float(observed[:, i].max()),
                     "trainingEncodedMean": float(offset[i]), "trainingEncodedScale": float(scale[i]),
                     "outsideTrainingRangeFraction": float(np.mean((observed[:, i] < low) | (observed[:, i] > high))),
                     "absoluteStandardizedP99": float(np.quantile(standardized[:, i], .99)),
                     "absoluteStandardizedMax": float(standardized[:, i].max()),
                     "trainingZeroFraction": float(np.mean(train[:, i] == 0)),
                     "observedZeroFraction": float(np.mean(observed[:, i] == 0))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("training", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, _ = load_dataset(args.dataset)
    run = json.loads((args.training / "summary.json").read_text())
    test = json.loads((args.test / "manifest.json").read_text())
    if (run["status"] != "complete" or test["status"] not in ("complete", "complete_with_exclusions")
            or run["dataset_manifest_sha256"] != sha256(args.dataset / "manifest.json")
            or test["species_names"] != physics["species_names"]):
        raise ValueError("Require complete and identity-matched training and test artifacts")
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("frozen-plan.json", "frozen_plan_sha256")):
        if sha256(args.test / name) != test[key]:
            raise ValueError("Test input identity mismatch")
    plan = json.loads((args.test / "frozen-plan.json").read_text())
    frozen = [entry for entry in plan["models"] if Path(entry["directory"]) == args.training.resolve()]
    if len(frozen) != 1:
        raise ValueError("Training run is not uniquely present in the frozen plan")
    for name, digest in frozen[0]["sha256"].items():
        if sha256(args.training / name) != digest:
            raise ValueError("A frozen training artifact changed")
    if {row["target"] for row in run["variants"]} != {"state-boxcox", "signed-power", "budget-linear", "scaled-asinh"}:
        raise ValueError("Require the primary four-target comparison")
    expected_offset, expected_scale = standardization(input_features(data["train"]["states"]))
    offset, scale, scaler_hashes = None, None, []
    for variant in run["variants"]:
        if variant["training_count"] != len(data["train"]["states"]):
            raise ValueError("This profile requires the complete accepted training set")
        path = args.training / variant["name"] / "preprocessing.npz"
        with np.load(path, allow_pickle=False) as arrays:
            current_offset, current_scale = arrays["x_offset"], arrays["x_scale"]
        np.testing.assert_allclose(current_offset, expected_offset, rtol=1e-12, atol=1e-30)
        np.testing.assert_allclose(current_scale, expected_scale, rtol=1e-12, atol=1e-30)
        if offset is not None:
            np.testing.assert_array_equal(current_offset, offset)
            np.testing.assert_array_equal(current_scale, scale)
        offset, scale = current_offset, current_scale
        scaler_hashes.append({"variant": variant["name"], "sha256": sha256(path)})
    with np.load(args.test / "labels.npz", allow_pickle=False) as arrays:
        accepted = arrays["accepted"]
    with np.load(args.test / "source-states.npz", allow_pickle=False) as arrays:
        populations = {"validation": data["validation"]["states"],
                       **{name: arrays["states"][accepted & arrays[name]] for name in ("uniform", "balanced")}}
    names = ["T_K", "P_Pa", *physics["species_names"]]
    rows = [row for name, states in populations.items() for row in
            profile_population(data["train"]["states"], states, offset, scale, names, name)]
    result = {"status": "complete", "source": source_revision(),
              "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
              "training_summary_sha256": sha256(args.training / "summary.json"),
              "test_manifest_sha256": sha256(args.test / "manifest.json"),
              "preprocessing": scaler_hashes, "rows": rows,
              "scope": "Post-score descriptive input-support check. No model changes or causal attribution."}
    if not args.dry_run:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"dry_run": args.dry_run, "features": len(names),
                      "populations": {name: len(states) for name, states in populations.items()},
                      "common_input_scaler_verified": True}))


if __name__ == "__main__":
    main()
