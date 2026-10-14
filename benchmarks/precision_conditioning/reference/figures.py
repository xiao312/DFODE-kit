"""Export source-backed static figures for checkpoint issue attachments."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import assess


def build(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(line) for line in (output / "intervals.jsonl").read_text().splitlines()]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    order = ["default", "relative10", "relative12", "absolute18", "absolute21", "step_limited", "radau_check", "fp32_input"]
    labels = ["1e-9 / 1e-15", "1e-10 / 1e-15", "1e-12 / 1e-15", "1e-12 / 1e-18", "1e-12 / 1e-21", "step limited", "Radau check", "FP32 input"]
    values = [summary["solvers"][name]["species_budget_error_max_per_interval"]["p99"] for name in order]
    positions = np.arange(len(order))
    axes[0].scatter(values, positions, color="#27628E", s=55)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].axvline(.01, color="#B66B12", linestyle="--", label="reference check: 0.01")
    axes[0].axvline(1, color="#303840", linestyle=":", label="physical budget: 1")
    axes[0].set_xlabel("p99 of per-interval maximum species budget error")
    axes[0].set_title("Solver settings and FP32 input effects")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(axis="x", alpha=.2)
    bins = summary["magnitude_bins"]
    counts = np.array([item["count"] for item in bins])
    budget = np.array([item["budget_fit_count"] for item in bins])
    relative = np.array([item["relative_fit_count"] for item in bins])
    x = np.arange(len(bins))
    axes[1].bar(x, counts - budget, bottom=budget, color="#D5DADF", label="budget unresolved")
    axes[1].bar(x, budget - relative, bottom=relative, color="#E1AC63", label="budget fit only")
    axes[1].bar(x, relative, color="#27628E", label="budget and relative fit")
    axes[1].set_xticks(x, ["< -32", "-32 to -24", "-24 to -16", "-16 to -8", ">= -8"])
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].set_xlabel("log10 absolute direct increment (zero estimates excluded)")
    axes[1].set_ylabel("Species components across intervals")
    axes[1].set_title("Label fitness by increment magnitude")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"Checkpoint 02 | {summary['interval_count']} H2 and CH4 intervals | No model training", fontsize=15)
    fig.savefig(output / "reference-feasibility.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for mechanism, color, marker in [("h2", "#27628E", "o"), ("ch4", "#B66B12", "x")]:
        x, y = [], []
        for record in records:
            if record["mechanism"] != mechanism:
                continue
            checked = assess(record, manifest["config"])
            if checked is None:
                continue
            active = np.abs(checked["reference"][1:]) > 0
            x.extend(np.abs(checked["reference"][1:][active]))
            y.extend((checked["uncertainty"][1:] / checked["weights"][1:])[active])
        axes[0].scatter(x, np.maximum(y, 1e-18), s=14, alpha=.5, color=color, marker=marker, label=mechanism.upper())
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].axhline(.01, color="#303840", linestyle="--")
    axes[0].set_xlabel("Absolute direct species increment")
    axes[0].set_ylabel("Estimated uncertainty / physical error budget")
    axes[0].set_title("Independent-check uncertainty")
    axes[0].legend()
    axes[0].text(.02, .02, "Display floor: 1e-18; not an uncertainty bound", transform=axes[0].transAxes, fontsize=9,
                 bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    species = sorted(summary["species"], key=lambda row: row["endpoint_zero_direct_nonzero_count"], reverse=True)[:12]
    axes[1].barh(np.arange(len(species)), [row["endpoint_zero_direct_nonzero_count"] for row in species], color="#27628E")
    axes[1].set_yticks(np.arange(len(species)), [f"{row['mechanism'].upper()} {row['species']}" for row in species])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Endpoint zero / direct nonzero count")
    axes[1].set_title("Largest counts by species (up to 12)")
    fig.suptitle("Cancellation diagnostics | Direct integration uses shared FP64 chemistry", fontsize=15)
    fig.savefig(output / "reference-cancellation.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"input": str(args.output), "figures": ["reference-feasibility.png", "reference-cancellation.png"]}))
    else:
        build(args.output)
