"""Static multi-seed fit evidence and explicit gates for increasing data size."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from review import table

NAMES = {"adam-decay": "Adam with decay", "lbfgs": "L-BFGS", "linear-head": "Direct final-layer solve"}
COLORS = {"adam-decay": "#27628E", "lbfgs": "#B66B12", "linear-head": "#26765D"}


def public_summary(summary):
    result = {key: value for key, value in summary.items() if key != "results"}
    result["results"] = []
    for row in summary["results"]:
        exported = {key: value for key, value in row.items() if key != "history"}
        exported["history_points"] = len(row["history"])
        exported["last_attempt"] = row["history"][-1]
        result["results"].append(exported)
    return result


def figure(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained", sharex=True)
    seeds = summary["config"]["seeds"]
    for i, mechanism in enumerate(("h2", "ch4")):
        for j, subset in enumerate(("one", "narrow", "all")):
            ax = axes[i, j]
            for position, method in enumerate(summary["config"]["methods"]):
                rows = [r for r in summary["results"] if r["mechanism"] == mechanism and r["subset"] == subset and r["method"] == method]
                values = [r["selected"]["max_budget"] for r in rows]
                ax.plot([min(values), max(values)], [position, position], color=COLORS[method], alpha=.35)
                for row in rows:
                    offset = (seeds.index(row["seed"]) - 1) * .12
                    ax.scatter(row["selected"]["max_budget"], position + offset, color=COLORS[method],
                               marker=("o", "s", "^")[seeds.index(row["seed"])], s=40)
            ax.set_yticks(range(3), [NAMES[m] if j == 0 else "" for m in summary["config"]["methods"]])
            ax.set_ylim(2.5, -.5)
            ax.set_xscale("log")
            ax.set_xlim(1e-13, 1e8)
            ax.set_xticks([1e-12, 1e-8, 1e-4, 1., 1e4, 1e8])
            ax.axvline(1, color="#343B44", linestyle=":")
            count = 1 if subset == "one" else 8 if subset == "narrow" else summary["datasets"][mechanism]["training_rows"]
            ax.set_title(f"{mechanism.upper()} | {count} training {'row' if count == 1 else 'rows'}")
            ax.grid(axis="x", alpha=.2)
            if i == 1:
                ax.set_xlabel("Maximum physical error / budget\nAcross temperature and all species")
    fig.suptitle("Selected, saved fits | Three seeds per method | Dotted line: budget limit = 1", fontsize=14)
    fig.savefig(output / "polish-results.png", dpi=170)
    plt.close(fig)


def report(summary):
    rows = summary["results"]
    grouped, details = [], []
    for subset in ("one", "narrow", "all"):
        for method in summary["config"]["methods"]:
            chosen = [r for r in rows if r["subset"] == subset and r["method"] == method]
            passed = sum(r["final"]["all_components_pass"] for r in chosen)
            maximums = [r["selected"]["max_budget"] for r in chosen]
            grouped.append([subset, NAMES[method], f"{passed} / {len(chosen)}", f"{min(maximums):.4g} to {max(maximums):.4g}"])
    for row in rows:
        details.append([row["mechanism"].upper(), row["subset"], row["seed"], NAMES[row["method"]],
                        f"{row['selected']['max_budget']:.5g}", row["stop_reason"],
                        f"{row['selected']['stage']} / {row['selected']['step']}", f"{row['elapsed_seconds']:.2f}"])
    results_table = table(["Training subset", "Method", "All-component passes", "Observed max-error range / budget"], grouped)
    detail_table = table(["Mechanism", "Subset", "Seed", "Method", "Maximum / budget", "Stop reason", "Saved stage / step", "Seconds"], details)
    provenance = escape(json.dumps({"source": summary["source"], "torch": summary["torch"],
                                   "datasets": summary["datasets"], "config": summary["config"]}, indent=2))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Repeatable fitting and the dataset expansion decision</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#182b3e;background:#f4f6fa;max-width:1200px;margin:auto;padding:28px}}
section{{background:white;padding:24px;margin:22px 0;border-radius:12px}}h1,h2{{line-height:1.25}}a{{color:#1d6094}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d6dfe8}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.notice{{border-left:5px solid #b66b12;padding:15px;background:#fff4e3}}
</style></head><body><nav><a href="../index.html">Research home</a> · <a href="../checkpoint-03-fit/report.html">Previous diagnosis</a> ·
<a href="summary.json">Measured results</a> · <a href="https://github.com/xiao312/DFODE-kit/issues/3">Review in Issue #3</a></nav>
<h1>Can we obtain a repeatable fit, and when should the dataset grow?</h1>
<p>Checkpoint 03 follow-up. All 54 fits ran on lh40902. No new chemistry labels or held-out scores were generated.</p>
<p class="notice"><strong>Partial milestone:</strong> every method now fits one example within every budget across both mechanisms and all three seeds.
The direct final-layer solve also passes all six eight-example cases. Neither iterative method passes those eight-example cases.
No method passes the full existing training sets. Repeatable narrow fitting and reliable saving are established;
a scalable general-purpose training recipe is not.</p>
<section><h2>Step 1 — Fix the acceptance rule before training</h2>
<p>The previous experiment sometimes reached accurate species predictions and then lost them with further updates.
We now check temperature AND every species after each accepted update. We stop at a maximum error of half a budget,
save the weights immediately, and independently replay them against the original full budget.</p>
<p>Species budget: <code>1e-12 + 1e-6 * abs(Y_initial)</code>. Temperature budget:
<code>1e-6 + 1e-8 * abs(T_initial)</code>. A maximum at or below 1 means every component passes.
The half-budget stop provides margin; it does not relax the evaluation threshold.</p>
<p>If a run does not pass, save its lowest-maximum training checkpoint and report failure.
This is declared training-based selection, not a test score. The saved step can differ from the last attempted step.</p></section>
<section><h2>Step 2 — Repeat three methods under the same conditions</h2>
<p>Use the same accepted checkpoint 03 data, budget-linear targets, training-only scales, FP64 and two 64-unit tanh layers.
For each mechanism, use one example, eight examples from its 1400 K training parent at h=1e-6 s,
and all existing training examples (H2: 133; CH4: 144). Repeat with seeds 20261008, 20261009 and 20261010.</p>
<p>Each method starts from the same initialization and up to 200 warm-up Adam updates. Then change only the refinement:</p>
<ul><li><strong>Adam with decay:</strong> reduce the step size from 1e-3 to 1e-7 over at most 5,000 more updates.</li>
<li><strong>L-BFGS:</strong> use gradient history and a line search to choose further steps, for at most 1,000 accepted steps.</li>
<li><strong>Direct final-layer solve:</strong> freeze the warmed-up hidden features and solve only the final coefficients by SVD least squares.</li></ul>
<p>All methods minimize the same normalized-coordinate squared error; physical maximum error controls saving/stopping.
This is not an equal-compute speed comparison. L-BFGS uses multiple function evaluations per accepted step.
The direct solve is still an interpolation control. The data, architecture, objective and stop rule were not changed after seeing results.</p></section>
<section><h2>Step 3 — Read the result</h2>
<img src="polish-results.png" alt="Three-seed maximum errors: all methods pass one example; only direct final-layer solves pass eight; none pass all training rows">
{results_table}
<p>Each table row contains six runs: two mechanisms times three seeds. The ranges are observed seed/mechanism variation,
not confidence intervals. Each dot in the figure is one saved model. Different marker shapes identify the three seeds.</p>
<p><strong>One example:</strong> all 18 runs pass. The iterative fits stop below 0.5 budgets and their saved weights replay correctly.
We no longer lose a passing fit by continuing to update it. Because stopping and refinement both changed from the old run,
this does not isolate the causal effect of step-size decay alone.</p>
<p><strong>Eight examples:</strong> all six direct solves pass. Their worst all-component error is about 1e-7 budgets.
The feature matrices have full row rank, but condition numbers near 1e5. Direct solving succeeds here;
the tested gradient-based procedures do not find an equally accurate solution within their limits.</p>
<p><strong>Existing full training sets:</strong> all 18 runs fail. The best maximum among these runs is still above 200,000 budgets.
This is a training-fit problem, not evidence of successful generalization. More data was not tested, so we cannot claim that it would never help.</p></section>
<section><h2>Step 4 — What the milestone does and does not establish</h2>
<p><strong>Achieved:</strong> deterministic all-component checks, immediate stopping, saved passing weights,
independent replay, and repeatable narrow fitting through a direct output-layer solve. That method passes the prescribed 12/12
one/eight-example gate across both mechanisms and all seeds.</p>
<p><strong>Not achieved:</strong> an iterative training method that passes the eight-example gate, a method that fits all existing rows,
or a model that predicts unseen chemistry states accurately. The original target of a scalable high-accuracy baseline remains open.</p>
<p>The head has 64 hidden features plus a bias per output. Eight rows can be interpolated with a full-rank feature matrix;
133 or 144 arbitrary row targets cannot generally be interpolated by adjusting only 65 coefficients per output.
The full neural network can change its hidden features too, so failure of the frozen-head control does not prove a capacity limit for the whole network.</p>
<p>Do not present eight-point fitting as a useful surrogate. It is also not evidence of many correct relative digits for every tiny increment.</p></section>
<section><h2>Step 5 — When should we use larger datasets?</h2>
<p><strong>Do not wait for perfect accuracy everywhere.</strong> But do not launch a large labeling campaign on the basis of this control-only pass.</p>
<ol><li><strong>Now: use the data we already have.</strong> Diagnose the fit on structured subsets of the 133/144 training rows.
Compare a well-conditioned output-layer fit or budget-aware fitting objective under a fixed compute cap.
Track both training error and capacity as the number of independent examples rises. Do not spend indefinitely repeating eight-point tests.</li>
<li><strong>Next: a small generalization pilot.</strong> Freeze the candidate recipe and choose new independent parent trajectories before generating intervals.
Use these as development validation, not a final test. Require useful improvement over zero change in both p99 and exceedance across parents,
finite predictions, and no unexplained conservation or positivity failures. This gate is weaker than solver replacement.</li>
<li><strong>Then: a data-size experiment.</strong> Propose roughly 1,000, 4,000 and 16,000 checked intervals per mechanism.
Use nested training sets, one fixed validation set and a sealed new test set. Add independent parents and targeted coverage;
do not inflate the count with adjacent samples alone. Record how many labels are rejected and why.</li>
<li><strong>Continue only while the evidence supports it.</strong> If held-out error improves as data increases, propose the next size.
If training remains poor, investigate fitting or capacity. If training is good but validation is poor, investigate coverage.
If a larger set gives little improvement, stop blind expansion and examine the error source.</li></ol>
<p>These sizes are planning targets, not launches made in this run. Before each increase, measure labeling/training cost and use approved compute placement.
Full-grid chemistry, NH3/CH4, residual stacks and full-trajectory tests remain separate decisions.</p></section>
<section><h2>Reproduction and limits</h2>
<p>The run completed in {summary['elapsed_seconds']:.2f} seconds with one CPU, 4 GB memory and no network/GPU.
No fit hit its 60-second limit. The global cap was 900 seconds. The kernel lacks a separate swap limit.</p>
<p>Cantera/CVODES supplied the original states; direct SciPy Radau supplied the previously checked labels.
Neither solver ran again here. Shared FP64 rate calculations and empirical reference-agreement checks remain limitations.</p>
<p>Raw data, complete histories, weights and preprocessing are under <code>runs/representation/polish-20261008</code> in the server project.
The public summary omits full histories but retains selected/final-attempt steps and measured scores.
<code>verify_polish.py --require-narrow linear-head</code> passes; the same gate for either iterative method fails.
All 54 selected models have been replayed independently. Code verification is not a scientific accuracy pass.</p>
<p><a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments/benchmarks/precision_conditioning/learning">Source and commands</a> ·
<a href="https://github.com/xiao312/DFODE-kit/pull/4">Draft PR #4</a></p>
<details><summary>All 54 measured results</summary>{detail_table}</details>
<details><summary>Source, input hashes and pinned settings</summary><pre>{provenance}</pre></details></section></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    if summary["status"] != "complete" or len(summary["results"]) != 54:
        raise ValueError("A complete polishing matrix is required")
    output = args.run / "review"
    print(json.dumps({"output": str(output)}))
    if not args.dry_run:
        output.mkdir(exist_ok=True)
        figure(summary, output)
        (output / "report.html").write_text(report(summary), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(public_summary(summary), indent=2, allow_nan=False))
