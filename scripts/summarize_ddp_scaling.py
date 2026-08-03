#!/usr/bin/env python3
"""Summarize fixed-global-batch DDP scaling logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dcu-counts", default="4,8,16,32")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def summarize(run_dir: Path, dcu_count: int) -> dict:
    records = load_records(run_dir / "train.log")
    start = next(item for item in records if item.get("event") == "training_start")
    epochs = [item for item in records if "epoch" in item]
    complete = next(
        item for item in reversed(records)
        if item.get("event") == "training_complete"
    )
    final = max(epochs, key=lambda item: int(item["epoch"]))
    first = min(epochs, key=lambda item: int(item["epoch"]))
    epoch_span = int(final["epoch"]) - int(first["epoch"])
    steady_seconds = (
        (float(final["elapsed_seconds"]) - float(first["elapsed_seconds"]))
        / epoch_span
        if epoch_span > 0
        else float(final["elapsed_seconds"])
    )
    train_samples = int(start["usable_train_samples"])
    total_epochs = int(final["epoch"])
    total_seconds = float(complete["elapsed_seconds"])
    best = min(epochs, key=lambda item: float(item["validation_loss"]))
    return {
        "dcu_count": dcu_count,
        "node_count": dcu_count // 4,
        "world_size": int(start["world_size"]),
        "batch_size_per_device": int(start["batch_size_per_device"]),
        "global_batch_size": int(start["global_batch_size"]),
        "epochs": total_epochs,
        "train_samples": train_samples,
        "total_seconds": total_seconds,
        "steady_seconds_per_epoch": steady_seconds,
        "steady_samples_per_second": train_samples / steady_seconds,
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "final_validation_loss": float(final["validation_loss"]),
        "final_train_loss": float(final["train_loss"]),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.benchmark_root)
    counts = [
        int(value)
        for value in args.dcu_counts.replace(":", ",").split(",")
    ]
    rows = [summarize(root / f"dcu-{count}", count) for count in counts]
    baseline = rows[0]
    for row in rows:
        speedup = (
            baseline["steady_seconds_per_epoch"]
            / row["steady_seconds_per_epoch"]
        )
        row["speedup_vs_4_dcu"] = speedup
        row["parallel_efficiency_vs_4_dcu"] = speedup / (
            row["dcu_count"] / baseline["dcu_count"]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_root": str(root),
        "comparison_policy": {
            "fixed_global_batch": baseline["global_batch_size"],
            "fixed_epochs": baseline["epochs"],
            "fixed_train_samples": baseline["train_samples"],
        },
        "results": rows,
    }
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
