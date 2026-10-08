"""Generate a bounded lineage-preserving flame dataset and CVODE labels."""
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
from benchmarks.flame_conditioning.augmentation import sample_split, validate_config
from benchmarks.flame_conditioning.chemistry import endpoint
from benchmarks.flame_conditioning.extract import sha256, source_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dataset.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    config = json.loads(args.config.read_text())
    source = np.load(args.source / "source-states.npz", allow_pickle=False)
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    validate_config(config, source["snapshot"])
    mechanism = args.source / "mechanism.yaml"
    if sha256(mechanism) != source_manifest["mechanism_sha256"] or sha256(args.source / "source-states.npz") != source_manifest["states_sha256"]:
        raise ValueError("Source artifact checksum mismatch")
    print(json.dumps({"config": config, "source": source_manifest["states_sha256"],
                      "test_policy": "No 2D test states are read by this command"}), flush=True)
    if args.dry_run:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(mechanism, args.output / "mechanism.yaml")
    started = time.monotonic()
    manifest = {
        "status": "running", "source": source_revision(), "config": config,
        "source_manifest": source_manifest, "cantera": ct.__version__, "splits": {},
        "deviations_from_paper": ["Bounded six-snapshot source rather than all time steps",
                                  "Pressure is interpolated, not randomly perturbed",
                                  "Argon is held at its interpolated source value",
                                  "No heat-release rejection filter; negative heat release is not universally invalid",
                                  "Independent species exponent draws and explicit non-argon normalization"],
    }

    def save():
        manifest["elapsed_seconds"] = time.monotonic() - started
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

    save()
    for split in ("train", "validation"):
        states, lineage, sampling = sample_split(source, source_manifest["species_names"], config, split)
        destination = args.output / split
        destination.mkdir()
        np.savez_compressed(destination / "inputs.npz", states=states, **lineage)
        delta = np.full((len(states), states.shape[1] - 2), np.nan)
        accepted = np.zeros(len(states), dtype=bool)
        report = {"sampling": sampling, "labels_completed": 0, "labels_accepted": 0, "failures": []}
        manifest["splits"][split] = report
        for index, row in enumerate(states):
            if time.monotonic() - started >= config["wall_seconds"]:
                manifest["status"] = "time_limit"
                break
            state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
            record = {"row": index}
            try:
                result = endpoint(mechanism, state, config["interval_s"], config["cvode_rtol"], config["cvode_atol"])
                delta[index] = result.pop("delta")
                record.update(result)
                physical = result["diagnostics"]
                if not np.isfinite(delta[index]).all() or physical["minimum_mass_fraction"] < 0:
                    raise ValueError("Nonfinite label or negative endpoint; retained but excluded")
                if abs(physical["mass_delta_sum"]) > 1e-10 or physical["element_delta_max"] > 1e-10:
                    raise ValueError("Label conservation failure")
                if abs(physical["temperature_change_K"]) > 1e-8 or abs(physical["relative_density_change"]) > 1e-10:
                    raise ValueError("Chemistry constraint failure")
                accepted[index] = True
            except Exception as error:
                record["error"] = str(error)
                report["failures"].append({"row": index, "error": str(error)})
            with (destination / "label-records.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            report["labels_completed"] = index + 1
            report["labels_accepted"] = int(accepted.sum())
            if index % 100 == 0:
                np.savez_compressed(destination / "labels.npz", delta=delta, accepted=accepted)
                save()
                print(json.dumps({"split": split, "completed": index + 1, "accepted": int(accepted.sum())}), flush=True)
        np.savez_compressed(destination / "labels.npz", delta=delta, accepted=accepted)
        report["inputs_sha256"] = sha256(destination / "inputs.npz")
        report["labels_sha256"] = sha256(destination / "labels.npz")
        save()
        if manifest["status"] == "time_limit":
            break
    if manifest["status"] == "running":
        manifest["status"] = "complete" if all(not report["failures"] for report in manifest["splits"].values()) else "complete_with_exclusions"
    save()


if __name__ == "__main__":
    main()
