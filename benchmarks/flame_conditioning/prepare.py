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
from benchmarks.flame_conditioning.chemistry import EndpointIntegrator
from benchmarks.flame_conditioning.extract import sha256, source_revision


def checkpoint_stride(row_count):
    """Bound full-array compression cost, with at most 1000 rows between saves."""
    return min(1000, max(100, row_count // 200))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dataset.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-source", type=Path)
    parser.add_argument("--continuation-wall-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new run directory")
    if not 1 <= args.continuation_wall_seconds <= 3600:
        parser.error("Continuation wall limit must be between 1 and 3600 seconds")
    config = json.loads(args.config.read_text())
    source = np.load(args.source / "source-states.npz", allow_pickle=False)
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    validate_config(config, source["snapshot"])
    mechanism = args.source / "mechanism.yaml"
    if sha256(mechanism) != source_manifest["mechanism_sha256"] or sha256(args.source / "source-states.npz") != source_manifest["states_sha256"]:
        raise ValueError("Source artifact checksum mismatch")
    previous = None
    if args.resume_source:
        previous = json.loads((args.resume_source / "manifest.json").read_text())
        if (previous["status"] != "time_limit" or previous["config"] != config
                or previous["source_manifest"] != source_manifest):
            raise ValueError("Resume requires a time-limited run with identical source and configuration")
        if sha256(args.resume_source / "mechanism.yaml") != source_manifest["mechanism_sha256"]:
            raise ValueError("Stopped mechanism checksum mismatch")
        for split, report in previous["splits"].items():
            for name in ("inputs", "labels"):
                if sha256(args.resume_source / split / f"{name}.npz") != report[f"{name}_sha256"]:
                    raise ValueError("Stopped dataset artifact checksum mismatch")
    print(json.dumps({"config": config, "source": source_manifest["states_sha256"],
                      "test_policy": "No 2D test states are read by this command"}), flush=True)
    if args.dry_run:
        return
    if args.resume_source:
        shutil.copytree(args.resume_source, args.output)
    else:
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
    previous_elapsed = 0
    execution_limit = config["wall_seconds"]
    if previous is not None:
        manifest = previous
        previous_elapsed = previous["elapsed_seconds"]
        manifest.setdefault("continuations", []).append({
            "previous_manifest_sha256": sha256(args.resume_source / "manifest.json"),
            "previous_source": previous["source"], "previous_elapsed_seconds": previous_elapsed,
            "wall_seconds": args.continuation_wall_seconds})
        manifest.update(status="running", source=source_revision())
        execution_limit = args.continuation_wall_seconds

    def save():
        manifest["elapsed_seconds"] = previous_elapsed + time.monotonic() - started
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

    save()
    integrator = EndpointIntegrator(mechanism, config["cvode_rtol"], config["cvode_atol"])
    for split in ("train", "validation"):
        destination = args.output / split
        if destination.exists():
            states = np.load(destination / "inputs.npz", allow_pickle=False)["states"]
            labels = np.load(destination / "labels.npz", allow_pickle=False)
            delta, accepted = labels["delta"].copy(), labels["accepted"].copy()
            report = manifest["splits"][split]
            if report["labels_accepted"] != int(accepted.sum()) or report["labels_completed"] > len(states):
                raise ValueError("Stopped label counters are inconsistent")
        else:
            states, lineage, sampling = sample_split(source, source_manifest["species_names"], config, split)
            destination.mkdir()
            np.savez_compressed(destination / "inputs.npz", states=states, **lineage)
            delta = np.full((len(states), states.shape[1] - 2), np.nan)
            accepted = np.zeros(len(states), dtype=bool)
            report = {"sampling": sampling, "labels_completed": 0, "labels_accepted": 0, "failures": []}
            manifest["splits"][split] = report
        report["checkpoint_rows"] = checkpoint_stride(len(states))
        for index in range(report["labels_completed"], len(states)):
            row = states[index]
            if time.monotonic() - started >= execution_limit:
                manifest["status"] = "time_limit"
                break
            state = {"T": float(row[0]), "P": float(row[1]), "Y": row[2:].tolist()}
            record = {"row": index}
            try:
                result = integrator.advance(state, config["interval_s"])
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
            if index % report["checkpoint_rows"] == 0:
                np.savez_compressed(destination / "labels.npz", delta=delta, accepted=accepted)
                save()
            if index % 100 == 0:
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
