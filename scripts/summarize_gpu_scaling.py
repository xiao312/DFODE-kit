#!/usr/bin/env python3
"""Summarize RTX fixed-global-batch scaling benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--gpu-counts", default="1,2,4,6")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def records(path: Path) -> list[dict]:
    values = []
    text = path.read_bytes().replace(b"\x00", b"").decode(
        "utf-8", errors="replace"
    )
    for line in text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def summarize(root: Path, gpu_count: int) -> dict:
    values = records(root / f"gpu-{gpu_count}" / "train.log")
    start = next(value for value in values if value.get("event") == "training_start")
    epochs = [value for value in values if "epoch" in value]
    complete = next(
        value
        for value in reversed(values)
        if value.get("event") == "training_complete"
    )
    first = min(epochs, key=lambda value: int(value["epoch"]))
    final = max(epochs, key=lambda value: int(value["epoch"]))
    span = int(final["epoch"]) - int(first["epoch"])
    steady = (
        (float(final["elapsed_seconds"]) - float(first["elapsed_seconds"]))
        / span
    )
    samples = int(start["usable_train_samples"])
    best = min(epochs, key=lambda value: float(value["validation_loss"]))
    return {
        "gpu_count": gpu_count,
        "batch_size_per_gpu": int(start["batch_size_per_device"]),
        "global_batch_size": int(start["global_batch_size"]),
        "train_samples": samples,
        "steady_seconds_per_epoch": steady,
        "steady_samples_per_second": samples / steady,
        "projected_2000_epoch_hours": steady * 2000.0 / 3600.0,
        "total_20_epoch_seconds": float(complete["elapsed_seconds"]),
        "best_validation_loss": float(best["validation_loss"]),
        "final_train_loss": float(final["train_loss"]),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.benchmark_root)
    counts = [int(value) for value in args.gpu_counts.split(",")]
    rows = [summarize(root, count) for count in counts]
    baseline = rows[0]
    for row in rows:
        speedup = (
            baseline["steady_seconds_per_epoch"]
            / row["steady_seconds_per_epoch"]
        )
        row["speedup_vs_1_gpu"] = speedup
        row["parallel_efficiency_vs_1_gpu"] = speedup / row["gpu_count"]

    payload = {"benchmark_root": str(root), "results": rows}
    output = Path(args.output)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with output.with_suffix(".csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
