"""Plot the synthetic audit for inline GitHub checkpoint review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in report["results"]}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    controls = ["endpoint-subtraction-fp64", "endpoint-storage-fp32-subtract-fp64"]
    counts = [by_name[name]["nonzero_predicted_zero"] for name in controls]
    bars = axes[0].bar(["FP64 endpoints", "FP32 endpoints"], counts, color=["#315f91", "#b66b38"])
    axes[0].bar_label(bars, padding=4)
    axes[0].set_ylim(0, max(counts) * 1.18)
    axes[0].set_ylabel("Known nonzero increments reconstructed as zero")
    axes[0].set_title("Endpoint subtraction loses information")

    coordinates = ["budget-linear", "budget-asinh", "signed-power"]
    positions = np.arange(3)
    for dtype, offset, color in [("float32", -0.18, "#b66b38"), ("float64", 0.18, "#315f91")]:
        values = [by_name[f"{name}-encode-{dtype}-decode-fp64"]["relative_error_p99"] for name in coordinates]
        axes[1].bar(positions + offset, values, width=0.35, label=dtype, color=color)
    axes[1].set_xticks(positions, ["Budget-linear", "Budget-asinh", "Signed-power"])
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-17, 1e-5)
    axes[1].set_ylabel("99th percentile relative round-trip error")
    axes[1].set_title("Coordinate dtype matters; all decode in FP64")
    axes[1].legend(title="Encoded coordinate")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(f"Synthetic representation audit · {report['sample_count']:,} pairs · no chemistry or model training")
    output = args.output or args.source.with_name("coordinate-audit.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
