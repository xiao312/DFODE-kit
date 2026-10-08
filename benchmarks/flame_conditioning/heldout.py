"""Freeze model evidence, sample a reserved CFD snapshot, and label fixed-T/V states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import EndpointIntegrator
from benchmarks.flame_conditioning.copy_case import MECHANISM_SHA256
from benchmarks.flame_conditioning.extract import sha256, source_revision

EDGES = [0, 305, 500, 1000, 1500, 2000, 3000]


def sample_cells(temperature, seed=20261009, uniform_count=1024, per_bin=32):
    temperature = np.asarray(temperature)
    if temperature.ndim != 1 or not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError("Temperature must be a finite positive vector")
    rng = np.random.default_rng(seed)
    uniform = rng.choice(len(temperature), min(uniform_count, len(temperature)), replace=False)
    balanced, populations = [], []
    for low, high in zip(EDGES[:-1], EDGES[1:], strict=True):
        ids = np.flatnonzero((temperature >= low) & (temperature < high))
        balanced.extend(rng.choice(ids, min(per_bin, len(ids)), replace=False).tolist())
        populations.append({"lower_K": low, "upper_K": high, "cells": len(ids)})
    selected = np.union1d(uniform, balanced).astype(int)
    return selected, np.isin(selected, uniform), np.isin(selected, balanced), populations


def normalize_rows(raw, species_count):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != species_count + 2 or not np.isfinite(raw).all():
        raise ValueError("Expected finite T,p,mechanism-ordered Y rows")
    if np.any(raw[:, :2] <= 0) or np.any(raw[:, 2:] < 0):
        raise ValueError("Invalid source state; do not clip or impute")
    mass_sum = raw[:, 2:].sum(axis=1)
    if np.max(np.abs(mass_sum - 1)) > 1e-4:
        raise ValueError("Source mass closure exceeds the recorded preprocessing limit")
    states = raw.copy()
    states[:, 2:] /= mass_sum[:, None]
    return states, mass_sum


def freeze_models(training_directories):
    plans = []
    for directory in training_directories:
        directory = directory.resolve()
        summary = json.loads((directory / "summary.json").read_text())
        config = summary["plan"]["config"]
        expected = len(config["targets"]) * len(config["precisions"]) * len(config["training_sizes"])
        if summary["status"] != "complete" or len(summary["variants"]) != expected:
            raise ValueError("Only complete pre-test comparisons may be frozen")
        files = {"summary.json": sha256(directory / "summary.json")}
        for variant in summary["variants"]:
            if variant["status"] != "complete":
                raise ValueError("Incomplete model")
            for filename in ("weights.pt", "preprocessing.npz", "result.json"):
                relative = f'{variant["name"]}/{filename}'
                files[relative] = sha256(directory / relative)
        plans.append({"directory": str(directory), "sha256": files})
    return plans


def freeze_historical(directories):
    plans = []
    for directory in directories:
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["status"] != "converted" or manifest["mechanism_sha256"] != MECHANISM_SHA256
                or sha256(directory / "model.npz") != manifest["arrays_sha256"]):
            raise ValueError("Historical numerical artifact is not a verified conversion")
        plans.append({"directory": str(directory), "kind": manifest["kind"],
                      "modes": ["source-formula", "stable-adapter"],
                      "sha256": {name: sha256(directory / name) for name in ("manifest.json", "model.npz")}})
    if len({item["kind"] for item in plans}) != len(plans):
        raise ValueError("Duplicate historical model kind")
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-array", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--historical", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if sha256(args.mechanism) != MECHANISM_SHA256:
        raise ValueError("Require the inspected 59-species study mechanism")
    frozen = freeze_models(args.training)
    plan = {"models": frozen, "historical": freeze_historical(args.historical),
            "seed": 20261009, "uniform_count": 1024, "per_temperature_bin": 32,
            "interval_s": 1e-6, "thresholds_K": [305, 1000], "source": source_revision()}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    # Commit comparison identities before opening the test snapshot.
    (args.output / "frozen-plan.json").write_text(json.dumps(plan, indent=2))
    started = time.monotonic()
    raw = np.load(args.source_array, mmap_mode="r", allow_pickle=False)
    gas = ct.Solution(str(args.mechanism))
    if raw.ndim != 2 or raw.shape[1] != gas.n_species + 2:
        raise ValueError("Reserved source does not have the verified T,p,Y schema")
    ids, uniform, balanced, populations = sample_cells(raw[:, 0])
    selected_raw = np.asarray(raw[ids])
    states, mass_sum = normalize_rows(selected_raw, gas.n_species)
    np.savez_compressed(args.output / "source-states.npz", states=states, raw_states=selected_raw,
                        sample_row=ids, snapshot=np.full(len(ids), "2D-0.002"),
                        uniform=uniform, balanced=balanced, original_mass_sum=mass_sum)
    shutil.copyfile(args.mechanism, args.output / "mechanism.yaml")
    manifest = {"status": "running", "source": source_revision(), "species_names": gas.species_names,
                "mechanism_sha256": MECHANISM_SHA256, "source_array_sha256": sha256(args.source_array),
                "frozen_plan_sha256": sha256(args.output / "frozen-plan.json"),
                "states_sha256": sha256(args.output / "source-states.npz"), "source_cells": len(raw),
                "temperature_populations": populations, "selected_cells": len(ids),
                "source_mass_closure_max": float(np.max(np.abs(mass_sum - 1))),
                "cantera": ct.__version__, "interval_s": 1e-6, "cvode_rtol": 1e-12, "cvode_atol": 1e-21,
                "labels_completed": 0, "labels_accepted": 0, "failures": []}
    delta = np.full((len(ids), gas.n_species), np.nan)
    accepted = np.zeros(len(ids), dtype=bool)
    integrator = EndpointIntegrator(args.output / "mechanism.yaml")

    def save():
        np.savez_compressed(args.output / "labels.npz", delta=delta, accepted=accepted)
        manifest["labels_sha256"] = sha256(args.output / "labels.npz")
        manifest["elapsed_seconds"] = time.monotonic() - started
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

    save()
    for index, row in enumerate(states):
        if time.monotonic() - started >= 600:
            manifest["status"] = "time_limit"
            break
        state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
        record = {"row": index, "cell": int(ids[index])}
        try:
            result = integrator.advance(state)
            delta[index] = result.pop("delta")
            record.update(result)
            check = result["diagnostics"]
            if (not np.isfinite(delta[index]).all() or check["minimum_mass_fraction"] < 0
                    or abs(check["mass_delta_sum"]) > 1e-10 or check["element_delta_max"] > 1e-10
                    or abs(check["temperature_change_K"]) > 1e-8 or abs(check["relative_density_change"]) > 1e-10):
                raise ValueError("Label fails positivity, conservation, or fixed-T/V constraints")
            accepted[index] = True
        except Exception as error:
            record["error"] = str(error)
            manifest["failures"].append({"row": index, "cell": int(ids[index]), "error": str(error)})
        with (args.output / "label-records.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        manifest.update(labels_completed=index + 1, labels_accepted=int(accepted.sum()))
        if index % 100 == 0:
            save()
            print(json.dumps({"completed": index + 1, "accepted": int(accepted.sum())}), flush=True)
    if manifest["status"] == "running":
        manifest["status"] = "complete_with_exclusions" if manifest["failures"] else "complete"
    save()


if __name__ == "__main__":
    main()
