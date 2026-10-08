"""Export measured comparison figures and a plain-language static review."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"budget-linear": "#27628E", "signed-power": "#B66B12", "scaled-asinh": "#79628F"}


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def figures(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, mechanism in zip(axes, ("h2", "ch4")):
        rows = [r for r in summary["variants"] if r["mechanism"] == mechanism]
        for position, row in enumerate(rows):
            ax.scatter(row["test"]["budget_p99"], position, color=COLORS[row["target"]],
                       marker="o" if row["precision"] == "float64" else "x", s=75)
        ax.set_yticks(range(len(rows)), [f"{r['target']} / {r['precision'].replace('float', 'FP')}" for r in rows])
        ax.invert_yaxis()
        ax.set_xscale("log")
        baseline = summary["datasets"][mechanism]["zero_baseline"]["budget_p99"]
        ax.axvline(baseline, color="#303840", linestyle="--", label="Predict zero change")
        ax.axvline(1, color="#303840", linestyle=":", label="Physical budget = 1")
        ax.set_xlabel("Test species error / physical budget, pooled p99")
        ax.set_title(f"{mechanism.upper()} | {summary['datasets'][mechanism]['counts']['test']} accepted test intervals")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("First representation comparison | Lower is better | One held-out parent per mechanism")
    fig.savefig(output / "comparison.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, mechanism in zip(axes, ("h2", "ch4")):
        for row in summary["variants"]:
            if row["mechanism"] != mechanism:
                continue
            ax.plot([c["epoch"] for c in row["curves"]], [c["validation_budget_p99"] for c in row["curves"]],
                    color=COLORS[row["target"]], linestyle="-" if row["precision"] == "float64" else "--",
                    label=f"{row['target']} / {row['precision'].replace('float', 'FP')}")
        ax.set_yscale("log")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Validation species budget error, pooled p99")
        ax.set_title(mechanism.upper())
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Validation selects the saved epoch | Test data do not select model settings")
    fig.savefig(output / "learning-curves.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, mechanism in zip(axes, ("h2", "ch4")):
        for row in summary["variants"]:
            if row["mechanism"] == mechanism and row["precision"] == "float64":
                bins = row["test"]["magnitude_bins"]
                ax.plot(range(5), [b.get("budget_p99", np.nan) for b in bins], marker="o", color=COLORS[row["target"]], label=row["target"])
        ax.set_xticks(range(5), ["< -32", "-32 to -24", "-24 to -16", "-16 to -8", ">= -8"])
        ax.tick_params(axis="x", labelsize=9)
        ax.set_yscale("log")
        ax.set_title(mechanism.upper())
        ax.set_xlabel("log10 absolute reference species change (zeros excluded)")
        ax.set_ylabel("Test species budget error, p99 within bin")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    fig.suptitle("FP64 models: errors by target magnitude | Empty bins have no point")
    fig.savefig(output / "magnitude-errors.png", dpi=170)
    plt.close(fig)


def report(summary):
    audits = summary["datasets"]
    rows = summary["variants"]
    rejected = sum(len(a["excluded"]) for a in audits.values())
    data_table = table(["Mechanism", "Checked intervals", "Train accepted", "Validation accepted", "Test accepted", "Excluded"],
                       [[name.upper(), a["source_rows"], *[a["counts"][s] for s in ("train", "validation", "test")], len(a["excluded"])] for name, a in audits.items()])
    measurements = table(["Mechanism", "Target", "Precision", "Saved epoch", "Test budget p99", "Above budget", "Train seconds"],
                         [[r["mechanism"].upper(), r["target"], r["precision"], r["selected_epoch"], f"{r['test']['budget_p99']:.5g}",
                           f"{100*r['test']['budget_exceedance']:.1f}%", f"{r['training_seconds']:.2f}"] for r in rows])
    decisions = []
    worst_rows = []
    for mechanism in audits:
        selected = min((r for r in rows if r["mechanism"] == mechanism), key=lambda r: r["validation_budget_p99"])
        baseline = audits[mechanism]["zero_baseline"]["budget_p99"]
        decisions.append(f'''<p><strong>{mechanism.upper()}:</strong> validation selected {escape(selected['target'])}, {selected['precision']}.
Its test p99 is {selected['test']['budget_p99']:.5g} budgets. Predicting zero gives {baseline:.5g} budgets.
Selection used validation data, not this test comparison.</p>''')
        species = sorted(enumerate(selected["test"]["species"]), key=lambda item: item[1]["budget_p99"], reverse=True)[:5]
        worst_rows.extend([mechanism.upper(), audits[mechanism]["species"][index], f"{values['budget_p99']:.5g}"] for index, values in species)
    worst = table(["Mechanism", "Species (validation-selected model)", "Test budget p99"], worst_rows)
    source = escape(json.dumps({"data_source": summary["provenance"]["pilot_source"], "training_source": summary["source"],
                               "raw_intervals_sha256": summary["provenance"]["intervals_sha256"], "versions": summary["provenance"]["versions"],
                               "torch": summary["torch"]}, indent=2))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Checkpoint 03: first learning comparison</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#182b3e;background:#f4f6fa;max-width:1150px;margin:auto;padding:28px}}
section{{background:white;padding:24px;margin:22px 0;border-radius:12px}}h1,h2{{line-height:1.2}}a{{color:#1d6094}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d6dfe8}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.notice{{border-left:5px solid #b66b12;padding:15px;background:#fff4e3}}
</style></head><body><nav><a href="../index.html">Research home</a> · <a href="summary.json">Complete measured metrics</a> ·
<a href="https://github.com/xiao312/DFODE-kit/issues/3">Review and decisions</a></nav>
<h1>Can a different target representation make a small model more accurate?</h1>
<p>This is our first chemistry learning test. All chemistry generation and model training ran on lh40902.
It is a small interpolation test, not a chemistry-solver replacement.</p>
<section><h2>Step 1 — Make checked answers</h2>
<p><strong>What we did:</strong> run eight H2 and eight CH4 reaction trajectories. Cantera/CVODES produced the starting states.
From each selected state, SciPy Radau integrated the temperature and species changes directly. It started the change at zero;
it did not obtain the label by subtracting two nearly equal stored endpoints.</p>
<p><strong>Why:</strong> the first pilot showed that endpoint subtraction can lose small changes. We compared the direct
answer with a second Radau setting and tight CVODES settings. Both methods still share Cantera FP64 reaction rates.
Agreement is evidence, not proof of exact answers.</p>
<p>We checked 384 intervals with nine solves per interval. Data generation took {summary['provenance']['data_seconds']:.1f} seconds.
We excluded {rejected} whole intervals because a required solve failed or a temperature/species component did not meet
the 1%-of-budget reference check. Excluded cases remain in the raw records.</p>{data_table}
<p><strong>Implication:</strong> the model results below apply to the accepted subset. Exclusion can remove difficult cases;
it does not prove that the model will work on those cases.</p></section>
<section><h2>Step 2 — Keep the test data separate</h2>
<p>Training parents start at 900, 1000, 1100, 1250, 1400 and 1500 K. Validation starts at 1150 K.
Testing starts at 1300 K. Each uses 1 atm and equivalence ratio 1. All intervals from one parent stay in one split.</p>
<p>Training data fit the model and normalization scales. Validation data select the saved epoch.
Test data measure the final result. No test score chooses an epoch or a transformation parameter.
There is only one validation and one test parent per mechanism. We cannot yet claim broad generalization.</p></section>
<section><h2>Step 3 — Change only the target coordinate</h2>
<p>Budget-linear divides the physical change by its error budget. Signed-power compresses the magnitude range.
Scaled asinh is nearly linear near zero and compresses larger values. We convert all predictions back to physical changes before scoring.</p>
<p>All variants use two 64-unit tanh layers, Adam at 0.001, seed 20261008 and 200 epochs. Each sees identical batches
and starts from identical numerical weights. Output channels that are always exactly zero in training are fixed to zero for all variants.
Each coordinate gets one FP32 and one FP64 model per mechanism: 12 fits, with no hyperparameter search.</p>
<p>FP32 here applies after FP64 input normalization. It is not the earlier test that rounded raw physical starting states.
Target normalization and final reconstruction use FP64. Timing includes validation; inference timing uses repeated batches and includes decoding.</p></section>
<section><h2>Step 4 — Compare physical errors</h2>
<p>The species budget is <code>1e-12 + 1e-6 * abs(Y_initial)</code>. Divide the prediction error by this budget.
A value below 1 is within budget. The reported p99 is the 99th percentile across accepted test species components;
it is not a guarantee on every sample and is not the interval-maximum statistic from checkpoint 02.</p>
<img src="comparison.png" alt="Test budget errors for all representations and precisions against the zero-change baseline">
{measurements}{''.join(decisions)}
<p>The zero baseline always predicts no chemical change. It is a useful check because many intervals have very small changes.
A model must improve on this baseline before we call it useful.</p>
<img src="magnitude-errors.png" alt="FP64 physical prediction errors grouped by reference increment magnitude">
<p>Magnitude bins exclude exact zero estimates. Counts and separately masked relative errors are in the downloadable metrics.
Relative errors use only components that passed the stricter reference-relative mask.</p>{worst}</section>
<section><h2>Step 5 — Check what the training curves support</h2>
<img src="learning-curves.png" alt="Validation budget errors during the fixed 200-epoch training runs">
<p>A lower error in one run does not establish a universal winner. One seed cannot measure training variability.
Different target coordinates and output scales also change the physical weighting of the training loss;
this is a comparison of complete target pipelines, not an isolated proof about numerical storage.</p>
<p class="notice">Do not infer an approximation ceiling from this single short training run. Before adding residual stages,
check whether the baseline fits training data, whether validation still improves, and whether the result repeats across parents and seeds.
No residual model or full-trajectory rollout has run in this checkpoint.</p></section>
<section><h2>Reproduction</h2><p>Raw states, labels, exclusion lists, saved weights, scales and test predictions remain in
<code>runs/representation/checkpoint03-20261008</code> under the server project. Runs use one CPU and no GPU.</p>
<p><a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments/benchmarks/precision_conditioning/learning">Source and instructions</a> ·
<a href="https://github.com/xiao312/DFODE-kit/pull/4">Draft PR</a></p><pre>{source}</pre></section></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.output / "training/summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("A complete matched comparison is required")
    destination = args.output / "review"
    print(json.dumps({"review": str(destination)}))
    if not args.dry_run:
        destination.mkdir(exist_ok=True)
        figures(summary, destination)
        (destination / "report.html").write_text(report(summary), encoding="utf-8")
        (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
