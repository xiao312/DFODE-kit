"""Hash-checked immutable chunks and final compatibility exports."""
import hashlib
import json
from pathlib import Path
import platform

import cantera as ct
import numpy as np

from benchmarks.flame_conditioning import chemistry, augmentation
from benchmarks.flame_conditioning.extract import sha256
from . import worker


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def contract(config, source_manifest):
    code = {name: sha256(Path(module.__file__)) for name, module in
            (("worker", worker), ("chemistry", chemistry), ("augmentation", augmentation))}
    for name in ("storage", "run"):
        code[name] = sha256(Path(__file__).with_name(name+".py"))
    value = dict(schema=1, config=config, source_manifest=source_manifest, code=code,
                 cantera=ct.__version__, numpy=np.__version__, python=platform.python_version())
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def commit_chunk(directory, start, delta, accepted, records):
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{start:09d}-{start+len(delta):09d}"
    path = directory / (name+".npz")
    record_path = directory / (name+".json")
    if path.exists() or record_path.exists():
        raise ValueError("Refuse overwriting a committed chunk")
    temporary = directory / (name+".tmp.npz")
    np.savez_compressed(temporary, delta=delta, accepted=accepted)
    temporary.replace(path)
    save_json(record_path, records)
    return dict(start=start, stop=start+len(delta), file=path.name, sha256=sha256(path),
                records=record_path.name, records_sha256=sha256(record_path),
                accepted=int(accepted.sum()))


def read_chunk(directory, entry, species_count):
    start, stop = entry["start"], entry["stop"]
    expected = f"{start:09d}-{stop:09d}"
    if start < 0 or stop <= start or entry["file"] != expected+".npz" or entry["records"] != expected+".json":
        raise ValueError("Invalid chunk identity")
    path, record_path = directory / entry["file"], directory / entry["records"]
    if sha256(path) != entry["sha256"] or sha256(record_path) != entry["records_sha256"]:
        raise ValueError("Chunk checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        delta, accepted = saved["delta"], saved["accepted"]
    if delta.shape != (stop-start, species_count) or accepted.shape != (stop-start,) or accepted.dtype != bool:
        raise ValueError("Invalid chunk shape")
    if int(accepted.sum()) != entry["accepted"] or not np.isfinite(delta[accepted]).all():
        raise ValueError("Invalid chunk accepted labels")
    records = json.loads(record_path.read_text())
    if [row["row"] for row in records] != list(range(start, stop)):
        raise ValueError("Chunk record row IDs differ")
    return delta, accepted, records


def coverage(count, entries):
    done = np.zeros(count, dtype=bool)
    for entry in entries:
        start, stop = entry["start"], entry["stop"]
        if not 0 <= start < stop <= count or done[start:stop].any():
            raise ValueError("Overlapping or out-of-range chunks")
        done[start:stop] = True
    return done


def missing_ranges(done, chunk_rows):
    ranges = []
    index = 0
    while index < len(done):
        if done[index]:
            index += 1
            continue
        stop = index+1
        while stop < len(done) and not done[stop] and stop-index < chunk_rows:
            stop += 1
        ranges.append((index, stop))
        index = stop
    return ranges


def export_split(directory, report, species_count):
    done = coverage(report["rows"], report["chunks"])
    if not done.all():
        raise ValueError("Cannot export an unfinished split")
    delta = np.full((len(done), species_count), np.nan)
    accepted = np.zeros(len(done), dtype=bool)
    failures = []
    with (directory / "label-records.jsonl").open("w") as handle:
        for entry in sorted(report["chunks"], key=lambda row: row["start"]):
            values, mask, records = read_chunk(directory / "chunks", entry, species_count)
            delta[entry["start"]:entry["stop"]] = values
            accepted[entry["start"]:entry["stop"]] = mask
            for record in records:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
                if "error" in record:
                    failures.append(dict(row=record["row"], error=record["error"]))
    np.savez_compressed(directory / "labels.npz", delta=delta, accepted=accepted)
    report.update(labels_completed=len(done), labels_accepted=int(accepted.sum()), failures=failures,
                  labels_sha256=sha256(directory / "labels.npz"))


def same_dataset_config(new, old):
    def numerical(config):
        return {key: value for key, value in config.items() if key not in ("train_count", "wall_seconds")}
    if numerical(new) != numerical(old) or new["train_count"] < old["train_count"]:
        raise ValueError("Reuse changed source domain, sampling or chemistry configuration")


def assert_prefix(inputs, previous):
    if set(inputs) != set(previous):
        raise ValueError("Reuse input fields differ")
    count = len(previous["states"])
    for key in inputs:
        if not np.array_equal(inputs[key][:count], previous[key]):
            raise ValueError(f"Reuse input prefix differs: {key}")
    return count
