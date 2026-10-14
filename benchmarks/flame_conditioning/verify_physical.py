"""Independently reconcile main validation claims with stored predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision


def recompute(states, prediction, reference, mechanism, interval):
    gas = ct.Solution(str(mechanism))
    gas.TP = 298.15, ct.one_atm
    enthalpy = gas.partial_molar_enthalpies / gas.molecular_weights
    active = np.array([name != "AR" for name in gas.species_names])
    density = []
    for row in states:
        gas.TPY = row[0], row[1], row[2:]
        density.append(gas.density)
    density = np.array(density)
    rate_error = np.sum((prediction - reference) * enthalpy, axis=1) * density / interval
    rate_reference = np.sum(reference * enthalpy, axis=1) * density / interval
    errors = np.abs(prediction[:, active] - reference[:, active])
    budgets = 1e-12 + 1e-6 * np.abs(states[:, 2:][:, active])
    negative = (states[:, 2:] + prediction)[:, active] < 0
    species_budget_p99 = {}
    for name in ("NH3", "CH4", "NO", "OH"):
        if name in gas.species_names:
            index = gas.species_index(name)
            error = np.abs(prediction[:, index] - reference[:, index])
            budget = 1e-12 + 1e-6 * np.abs(states[:, index + 2])
            species_budget_p99[name] = float(np.percentile(error / budget, 99))
    temperature_bins = []
    boundaries = [0, 305, 500, 1000, 1500, 2000, 3000]
    for index in range(len(boundaries) - 1):
        low, high = boundaries[index:index + 2]
        mask = (states[:, 0] >= low) & (states[:, 0] < high)
        values = (errors / budgets)[mask]
        temperature_bins.append({"lower_K": low, "upper_K": high, "samples": int(mask.sum()),
                                 "count": values.size, "p99": float(np.percentile(values, 99)) if values.size else None})
    magnitude_bins = []
    boundaries = [0, 1e-30, 1e-20, 1e-15, 1e-10, 1e-5, np.inf]
    for index in range(len(boundaries) - 1):
        low, high = boundaries[index:index + 2]
        mask = (np.abs(reference[:, active]) >= low) & (np.abs(reference[:, active]) < high)
        values = errors[mask]
        magnitude_bins.append({"lower": low, "upper": high if np.isfinite(high) else None,
                               "count": values.size, "p99": float(np.percentile(values, 99)) if values.size else None})
    return {"samples": len(states), "budget_p99": float(np.percentile(errors / budgets, 99)),
            "species_budget_p99": species_budget_p99,
            "temperature_bins": temperature_bins, "magnitude_bins": magnitude_bins,
            "negative_endpoint_fraction": float(negative.sum() / negative.size),
            "mass_drift_p99": float(np.percentile(np.abs(np.sum(prediction, axis=1)), 99)),
            "heat_error_rms": float(np.linalg.norm(rate_error) / np.sqrt(len(states))),
            "heat_reference_rms": float(np.linalg.norm(rate_reference) / np.sqrt(len(states)))}


def assert_scores(actual, recorded):
    expected = {"samples": recorded["samples"], "budget_p99": recorded["budget_error"]["p99"],
                "negative_endpoint_fraction": recorded["negative_endpoint_fraction"],
                "mass_drift_p99": recorded["mass_increment_drift"]["p99"],
                "heat_error_rms": recorded["heat_release_error_rms_W_m3"],
                "heat_reference_rms": recorded["heat_release_reference_rms_W_m3"]}
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=1e-30,
                                   err_msg=f"Saved physical metric differs: {key}")
    for name, value in actual["species_budget_p99"].items():
        np.testing.assert_allclose(value, recorded["per_species"][name]["budget_error"]["p99"],
                                   rtol=2e-12, atol=1e-30, err_msg=f"Saved species p99 differs: {name}")
    for kind, metric, identity in (("temperature_bins", "budget_error", ("lower_K", "upper_K", "samples")),
                                   ("magnitude_bins", "absolute_error", ("lower", "upper"))):
        if len(actual[kind]) != len(recorded[kind]):
            raise AssertionError(f"Saved {kind} length differs")
        for fresh, saved in zip(actual[kind], recorded[kind]):
            if any(fresh[key] != saved[key] for key in identity) or fresh["count"] != saved[metric]["count"]:
                raise AssertionError(f"Saved {kind} identities/counts differ")
            if fresh["p99"] is None or saved[metric]["p99"] is None:
                if fresh["p99"] is not saved[metric]["p99"]:
                    raise AssertionError(f"Saved {kind} empty-bin metric differs")
            else:
                np.testing.assert_allclose(fresh["p99"], saved[metric]["p99"], rtol=2e-12, atol=0,
                                           err_msg=f"Saved {kind} p99 differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--training", type=Path, action="append", default=[])
    parser.add_argument("--historical", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    if not args.training and not args.historical:
        parser.error("Supply at least one comparison")
    data, _, manifest = load_dataset(args.dataset)
    validation = data["validation"]
    records = []
    for directory, historical in [(p, False) for p in args.training] + [(p, True) for p in args.historical]:
        saved = json.loads((directory / "summary.json").read_text())
        if saved["status"] != "complete" or saved["dataset_manifest_sha256"] != sha256(args.dataset / "manifest.json"):
            raise ValueError("Require complete results on this exact dataset")
        for model in saved["models" if historical else "variants"]:
            path = directory / (f'{model["name"]}.npz' if historical else f'{model["name"]}/validation-predictions.npz')
            with np.load(path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["source_indices"], validation["source_indices"])
                actual = recompute(validation["states"], arrays["prediction"], validation["delta"],
                                   args.dataset / "mechanism.yaml", manifest["config"]["interval_s"])
            assert_scores(actual, model["validation"])
            records.append({"name": f'{directory.name}/{model["name"]}', "prediction_sha256": sha256(path), **actual})
    result = {"status": "verified", "source": source_revision(), "models": len(records), "records": records,
              "scope": "Main validation physical metrics, not full-trajectory accuracy",
              "dataset_manifest_sha256": sha256(args.dataset / "manifest.json")}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
