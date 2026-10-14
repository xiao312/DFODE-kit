"""Discover by default; generate or resume immutable CPU reference chunks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

# Spawned workers import NumPy/Cantera only after these limits are set.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import cantera as ct
import numpy as np

from benchmarks.flame_conditioning.augmentation import sample_split, validate_config
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from .storage import (assert_prefix, commit_chunk, contract, coverage, export_split,
                      missing_ranges, read_chunk, same_dataset_config, save_json)
from .worker import chunks


def preflight(source, config, output, resume=None, reuse=None):
    if output.exists():
        raise ValueError("Output exists; select a new directory")
    if resume and reuse:
        raise ValueError("Choose either resume or reuse")
    manifest = json.loads((source / "manifest.json").read_text())
    if (sha256(source / "mechanism.yaml") != manifest["mechanism_sha256"] or
            sha256(source / "source-states.npz") != manifest["states_sha256"]):
        raise ValueError("Source checksum mismatch")
    with np.load(source / "source-states.npz", allow_pickle=False) as states:
        validate_config(config, states["snapshot"])
    identity = contract(config, manifest)
    previous = None
    if resume:
        previous = json.loads((resume / "manifest.json").read_text())
        if previous["status"] not in ("running", "time_limit", "failed") or previous["contract"] != identity:
            raise ValueError("Resume needs an incomplete run with identical code/runtime/source/config")
        if sha256(resume / "mechanism.yaml") != manifest["mechanism_sha256"]:
            raise ValueError("Resume mechanism checksum mismatch")
        for split, report in previous["splits"].items():
            if sha256(resume / split / "inputs.npz") != report["inputs_sha256"]:
                raise ValueError("Resume input checksum mismatch")
            coverage(report["rows"], report["chunks"])
            for entry in report["chunks"]:
                read_chunk(resume / split / "chunks", entry, len(manifest["species_names"]))
    if reuse:
        _, _, old = load_dataset(reuse)
        same_dataset_config(config, old["config"])
        if old["source_manifest"] != manifest or old["cantera"] != ct.__version__:
            raise ValueError("Reuse source or Cantera version differs")
    return manifest, identity, previous


def import_labels(reuse, output, split, report):
    """Reuse both accepted and excluded labels, preserving row-level solver records."""
    with np.load(output / split / "inputs.npz", allow_pickle=False) as new:
        with np.load(reuse / split / "inputs.npz", allow_pickle=False) as old:
            count = assert_prefix(new, old)
    with np.load(reuse / split / "labels.npz", allow_pickle=False) as labels:
        delta, accepted = labels["delta"], labels["accepted"]
    records = [json.loads(line) for line in (reuse / split / "label-records.jsonl").read_text().splitlines()]
    if len(delta) != count or [row["row"] for row in records] != list(range(count)):
        raise ValueError("Reuse label records incomplete or out of order")
    entry = commit_chunk(output / split / "chunks", 0, delta, accepted, records)
    report["chunks"].append(entry)
    report["reused_rows"] = count


def execute(source, config, output, workers, chunk_rows, resume=None, reuse=None):
    source_manifest, identity, previous = preflight(source, config, output, resume, reuse)
    started = time.monotonic()
    deadline = started+config["wall_seconds"]
    output.mkdir(parents=True)
    shutil.copyfile(source / "mechanism.yaml", output / "mechanism.yaml")
    manifest = dict(status="running", source=source_revision(), source_manifest=source_manifest,
                    config=config, cantera=ct.__version__, contract=identity, splits={},
                    execution=dict(workers=workers, chunk_rows=chunk_rows, numerical_threads=1),
                    reused_dataset_manifest_sha256=sha256(reuse / "manifest.json") if reuse else None)
    if previous:
        manifest["splits"] = previous["splits"]
        manifest["resume_manifest_sha256"] = sha256(resume / "manifest.json")
        manifest["reused_dataset_manifest_sha256"] = previous.get("reused_dataset_manifest_sha256")
        for split, report in manifest["splits"].items():
            directory = output / split
            (directory / "chunks").mkdir(parents=True)
            shutil.copyfile(resume / split / "inputs.npz", directory / "inputs.npz")
            for entry in report["chunks"]:
                for key in ("file", "records"):
                    shutil.copyfile(resume / split / "chunks" / entry[key], directory / "chunks" / entry[key])

    def save():
        manifest["elapsed_seconds"] = time.monotonic()-started
        save_json(output / "manifest.json", manifest)

    save()
    try:
        with np.load(source / "source-states.npz", allow_pickle=False) as source_states:
            for split in ("train", "validation"):
                directory = output / split
                if split not in manifest["splits"]:
                    states, lineage, sampling = sample_split(source_states, source_manifest["species_names"], config, split)
                    directory.mkdir(exist_ok=True)
                    np.savez_compressed(directory / "inputs.npz", states=states, **lineage)
                    report = dict(rows=len(states), sampling=sampling, chunks=[], reused_rows=0,
                                  inputs_sha256=sha256(directory / "inputs.npz"))
                    if reuse:
                        import_labels(reuse, output, split, report)
                    manifest["splits"][split] = report
                    save()
                else:
                    with np.load(directory / "inputs.npz", allow_pickle=False) as saved:
                        states = saved["states"]
                    report = manifest["splits"][split]
                done = coverage(len(states), report["chunks"])
                ranges = missing_ranges(done, chunk_rows)
                labeling_started = time.monotonic()
                rows_new = 0
                for start, delta, accepted, records in chunks(output / "mechanism.yaml", config, states,
                                                             ranges, workers, deadline):
                    entry = commit_chunk(directory / "chunks", start, delta, accepted, records)
                    report["chunks"].append(entry)
                    rows_new += len(delta)
                    report["labels_completed"] = int(done.sum())+rows_new
                    report["labels_accepted"] = sum(item["accepted"] for item in report["chunks"])
                    report["new_rows_this_execution"] = rows_new
                    report["labeling_seconds_this_execution"] = time.monotonic()-labeling_started
                    save()
                    print(json.dumps(dict(split=split, completed=report["labels_completed"],
                                          accepted=report["labels_accepted"])), flush=True)
                export_split(directory, report, len(source_manifest["species_names"]))
                save()
        manifest["status"] = ("complete_with_exclusions" if any(report["failures"] for report in manifest["splits"].values())
                              else "complete")
        save()
        load_dataset(output)  # Compatibility and physical data gate before success.
    except TimeoutError:
        manifest["status"] = "time_limit"
        save()
        raise
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        save()
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    for name in ("config", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--resume-source", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chunk_rows <= 2048:
        parser.error("Chunk size must be 1 to 2048")
    config = json.loads(args.config.read_text())
    preflight(args.source, config, args.output, args.resume_source, args.reuse)
    print(json.dumps(dict(config=config, workers=args.workers, chunk_rows=args.chunk_rows,
                          execute=args.execute, test_policy="No test states are read")), flush=True)
    if args.execute:
        execute(args.source, config, args.output, args.workers, args.chunk_rows, args.resume_source, args.reuse)


if __name__ == "__main__":
    main()
