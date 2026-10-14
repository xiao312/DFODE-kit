"""Measure the historical enthalpy-curation rule without changing any dataset."""
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
from benchmarks.flame_conditioning.metrics import physical_scores


def curation_mask(delta, formation_enthalpies):
    energy_change = np.asarray(delta) @ np.asarray(formation_enthalpies)
    if not np.isfinite(energy_change).all():
        raise ValueError("Nonfinite formation-enthalpy change")
    return energy_change <= 200.0, energy_change


def population_record(states, kept, energy):
    return {"states": len(states), "kept": int(kept.sum()), "rejected": int((~kept).sum()),
            "rejected_fraction": float((~kept).mean()) if len(kept) else None,
            "endothermic_fraction": float((energy > 0).mean()) if len(energy) else None,
            "energy_change_min_J_kg": float(energy.min()) if len(energy) else None,
            "energy_change_max_J_kg": float(energy.max()) if len(energy) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--training", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, _ = load_dataset(args.dataset)
    dataset_hash = sha256(args.dataset / "manifest.json")
    runs = []
    for path in args.training:
        saved = json.loads((path / "summary.json").read_text())
        if saved["status"] != "complete" or saved["dataset_manifest_sha256"] != dataset_hash:
            raise ValueError("Only complete comparisons on this dataset may be audited")
        runs.append((path, saved))
    if args.dry_run:
        print(json.dumps({"dataset_manifest_sha256": dataset_hash, "training_runs": len(runs),
                          "rule": "sum(hf_298 * delta_Y) <= 200 J/kg", "no_mutation": True}))
        return
    result = {"status": "complete", "source": source_revision(), "dataset_manifest_sha256": dataset_hash,
              "rule": "sum(hf_298 * delta_Y) <= 200 J/kg", "splits": {}, "validation_models": [],
              "scope": "Curation sensitivity only; endothermic chemistry is not inherently an invalid reference"}
    masks = {}
    for split, values in data.items():
        kept, energy = curation_mask(values["delta"], physics["formation_enthalpies"])
        masks[split] = kept
        record = population_record(values["states"], kept, energy)
        record["temperature_bins"] = []
        edges = [0, 305, 500, 1000, 1500, 2000, 3000]
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            selected = (values["states"][:, 0] >= low) & (values["states"][:, 0] < high)
            record["temperature_bins"].append({"lower_K": low, "upper_K": high,
                **population_record(values["states"][selected], kept[selected], energy[selected])})
        result["splits"][split] = record
    validation = data["validation"]
    for directory, saved in runs:
        for model in saved["variants"]:
            path = directory / model["name"] / "validation-predictions.npz"
            with np.load(path, allow_pickle=False) as prediction:
                np.testing.assert_array_equal(prediction["source_indices"], validation["source_indices"])
                entry = {"name": f'{directory.name}/{model["name"]}', "prediction_sha256": sha256(path), "populations": {}}
                for name, mask in (("study-rule-kept", masks["validation"]), ("study-rule-rejected", ~masks["validation"])):
                    if mask.any():
                        entry["populations"][name] = physical_scores(prediction["prediction"][mask], validation["delta"][mask],
                            validation["states"][mask], prediction["correction"][mask], **physics)
            result["validation_models"].append(entry)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({"status": "complete", "splits": {name: {key: value for key, value in record.items() if key != "temperature_bins"}
                                                       for name, record in result["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
