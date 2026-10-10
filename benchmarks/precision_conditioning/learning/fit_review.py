"""Render the measured fit diagnosis for the existing static review site."""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from review import table

COLORS = {"coordinate": "#27628E", "physical-budget": "#B66B12", "head-control": "#26765D"}
NAMES = {"coordinate": "Adam: coordinate loss", "physical-budget": "Adam: physical loss", "head-control": "Direct linear-head solve"}


def maximum(scores):
    return max(scores["budget_max"], scores["temperature_budget_max"])


def figures(summary, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for ax, mechanism in zip(axes, ("h2", "ch4")):
        rows = [r for r in summary["results"] if r["mechanism"] == mechanism]
        for position, row in enumerate(rows):
            value = maximum(row["final"])
            color = COLORS[row["loss"]]
            if row["snapshots"]:
                earlier = maximum(row["snapshots"][0]["scores"])
                ax.plot([earlier, value], [position, position], color=color, alpha=.4)
                ax.scatter(earlier, position, facecolors="none", edgecolors=color, marker="o", s=55)
            ax.scatter(value, position, color=color, marker="D" if row["method"] == "linear-head-svd" else "o", s=45)
        ax.set_yticks(range(len(rows)), [f"{r['subset']} ({r['rows']} {'row' if r['rows'] == 1 else 'rows'}) | {NAMES[r['loss']]}" for r in rows])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1e-13, 1e9)
        ax.axvline(1, linestyle=":", color="#343B44", label="All-component pass limit = 1")
        ax.set_xlabel("Maximum error / budget across temperature and species\nLower is better; logarithmic scale")
        ax.set_title(mechanism.upper())
        ax.grid(axis="x", alpha=.2)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Training fit only | Open circle: 200 updates; filled circle: 5,000; diamond: direct head solve", fontsize=12)
    fig.savefig(output / "fit-errors.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for column, mechanism in enumerate(("h2", "ch4")):
        for row_index, subset in enumerate(("one", "narrow")):
            ax = axes[row_index, column]
            for row in summary["results"]:
                if row["mechanism"] == mechanism and row["subset"] == subset and row["method"] == "adam":
                    ax.plot([c["updates"] for c in row["curves"]], [c["species_p99"] for c in row["curves"]],
                            color=COLORS[row["loss"]], label=NAMES[row["loss"]])
            ax.axhline(1, color="#343B44", linestyle=":", label="Budget = 1 (p99 does not bound max)")
            ax.set_yscale("log")
            ax.set_xlabel("Adam updates; fixed learning rate 0.001")
            ax.set_ylabel("Training species budget error, pooled p99")
            ax.set_title(f"{mechanism.upper()} | {'one example' if subset == 'one' else 'eight examples'}")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("More updates do not guarantee a better final fit | Panels have separate vertical ranges")
    fig.savefig(output / "fit-curves.png", dpi=170)
    plt.close(fig)


def report(summary):
    rows = summary["results"]
    adam = [r for r in rows if r["method"] == "adam"]
    controls = [r for r in rows if r["method"] == "linear-head-svd"]
    passed_adam = sum(r["final"]["all_components_pass"] for r in adam)
    passed_controls = sum(r["final"]["all_components_pass"] for r in controls)
    measures = table(["Mechanism", "Subset / rows", "Loss", "Species p99 at 200", "Species p99 at 5,000", "Species RMS at 5,000", "Species above budget", "Max: T and species"],
                     [[r["mechanism"].upper(), f"{r['subset']} / {r['rows']}", r["loss"],
                       f"{r['snapshots'][0]['scores']['budget_p99']:.5g}", f"{r['final']['budget_p99']:.5g}",
                       f"{r['final']['budget_rms']:.5g}", f"{100*r['final']['budget_exceedance']:.1f}%", f"{maximum(r['final']):.5g}"] for r in adam])
    capacity = table(["Mechanism", "Subset / rows", "Feature rank", "Condition number", "Max: T and species", "Within budget?"],
                     [[r["mechanism"].upper(), f"{r['subset']} / {r['rows']}", r["feature_rank"], f"{r['feature_condition']:.5g}",
                       f"{maximum(r['final']):.5g}", "yes" if r["final"]["all_components_pass"] else "no"] for r in controls])
    roundtrip = table(["Mechanism", "Encode-normalize-decode maximum / budget"],
                      [[name.upper(), f"{maximum(value['roundtrip']):.5g}"] for name, value in summary["datasets"].items()])
    intermediate = []
    for row in adam:
        if row["subset"] == "one":
            best = min(row["curves"], key=lambda point: point["species_max"])
            intermediate.append([row["mechanism"].upper(), row["loss"], best["updates"], f"{best['species_max']:.5g}"])
    intermediate_table = table(["Mechanism", "One-example loss", "Lowest recorded species max: update", "Species max / budget"], intermediate)
    source = escape(json.dumps({"source": summary["source"], "input_run": summary["input_run"],
                               "input_sha256": {name: d["input_sha256"] for name, d in summary["datasets"].items()},
                               "torch": summary["torch"], "config": summary["plan"]["config"]}, indent=2))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Checkpoint 03 follow-up: can the model fit known answers?</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#182b3e;background:#f4f6fa;max-width:1200px;margin:auto;padding:28px}}
section{{background:white;padding:24px;margin:22px 0;border-radius:12px}}h1,h2{{line-height:1.25}}a{{color:#1d6094}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d6dfe8}}
.scroll{{overflow-x:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.notice{{border-left:5px solid #b66b12;padding:15px;background:#fff4e3}}
</style></head><body><nav><a href="../index.html">Research home</a> · <a href="../checkpoint-03/report.html">First model comparison</a> ·
<a href="summary.json">Measured results</a> · <a href="https://github.com/xiao312/DFODE-kit/issues/3">Review in Issue #3</a></nav>
<h1>Can the model fit answers it has already seen?</h1>
<p>Checkpoint 03 follow-up. All fitting ran on lh40902. This is a training-data diagnosis, not a new test of unseen states.</p>
<p class="notice"><strong>Main result:</strong> {passed_controls} of {len(controls)} direct linear-head controls meet every physical budget.
{passed_adam} of {len(adam)} Adam fits meet every budget at the fixed final checkpoint of 5,000 updates.
Three one-example Adam runs briefly meet the species budgets, then lose that accuracy.
The narrow examples can be represented, but the tested training procedure does not reliably keep an accurate fit.
A loss change alone does not solve the problem.</p>
<section><h2>Step 1 — Reproduce the problem</h2>
<p>We loaded the previous H2 budget-linear FP64 model and asked it to predict the first training example again.
The largest species error was 2.146 million times its budget. The read-only <code>fit_probe.py --row 0</code> check returned failure.
This showed that poor results were not limited to new, unseen states.</p>
<p><strong>Why this matters:</strong> before asking whether the model generalizes, first check whether the fitting method can reproduce known answers.
A failed scientific accuracy threshold does not, by itself, identify a software defect.</p></section>
<section><h2>Step 2 — Use checked answers, without new chemistry solves</h2>
<p>We reused the accepted checkpoint 03 reference data. Cantera/CVODES had supplied the starting states.
SciPy Radau had integrated the changes directly, with the reference-agreement checks described in the earlier reports.
Neither solver generated new labels in this diagnosis. These labels remain working references, not exact mathematical truth.</p>
<p>For each mechanism, we used one example, eight examples, and all training examples (H2: 133; CH4: 144).
The eight examples come from the 1400 K training parent, at eight time anchors, with interval h = 1e-6 s.
The one-example set is its first anchor. A 1400 K parent does not mean the temperature stays at 1400 K during reaction.</p>
<p>We did not use validation or test rows. Scales use only the original training split, and stay fixed across subsets.
The old test set has already been reviewed; it must not silently become a tuning set.</p></section>
<section><h2>Step 3 — Separate three questions</h2>
<ol><li><strong>Loss weighting:</strong> for the same subset and model, compare the existing normalized-coordinate loss with physical-budget loss.</li>
<li><strong>Training duration:</strong> compare fixed checkpoints at 200 and 5,000 updates in each run. Do not select the best intermediate result.</li>
<li><strong>Range of examples:</strong> compare one, eight and all training rows. This changes the task, so it is a separate probe.</li></ol>
<p>All fits use budget-linear outputs, FP64, two 64-unit tanh layers, seed 20261008, Adam at learning rate 0.001, and full batches.
The same mechanism starts from the same weights in every run. The original 200-update study selected an epoch using validation;
these fixed final checkpoints are not a direct replacement for those selected-model scores.</p>
<p><strong>What the two losses mean:</strong> the existing loss divides each output by its typical training magnitude.
It asks for similar accuracy relative to that typical magnitude. The physical loss instead gives equal weight to equal fractions of the specified error budgets.
We divide that physical loss by one global training-only constant to keep the numbers manageable. Relative weights stay unchanged;
Adam's effective epsilon can still change. Both losses include temperature and active species outputs.</p>
<p>Neither loss directly minimizes the worst error or p99. Average error can improve while more small components leave their budgets.</p></section>
<section><h2>Step 4 — Read the measured errors</h2>
<p>Species budget: <code>1e-12 + 1e-6 * abs(Y_initial)</code>. Temperature budget: <code>1e-6 + 1e-8 * abs(T_initial)</code>.
Divide absolute error by its budget. A value of 1 is the limit. The strict pass requires every component to be at or below 1.</p>
<img src="fit-errors.png" alt="Maximum training errors for 200 and 5000 Adam updates and direct linear-head controls; only direct controls meet all budgets">
{measures}
<p>Species p99 pools every species component in the selected rows. H2 has 10 species; CH4 has 53.
For example, the eight-row sets contain 80 and 424 species errors. Temperature is not in species p99; it is in the all-component maximum.</p>
<p><strong>What changed:</strong> more updates greatly reduce the full-set error, but do not reach the budgets.
For CH4 on all training rows, physical loss reduces final species p99 from about 21,850 to 14,754 budgets.
However, the fraction above budget rises from about 81% to 94%. For H2, physical loss slightly worsens p99.
These results do not support a universal loss winner.</p>
<img src="fit-curves.png" alt="One-example and eight-example Adam training curves show non-monotonic physical errors">
<p><strong>Important limit:</strong> the one-example coordinate fits finish worse at 5,000 updates than at 200.
These curves show that more updates at a fixed step size are not a reliable route to high accuracy.
They do not, on their own, identify which optimizer setting should replace it.</p>
{intermediate_table}
<p>Three one-example runs briefly bring every species error within budget, then move out again.
This is a post-run diagnosis of the recorded curve, not a new selected-model result.
Intermediate temperature maxima and model weights were not saved, so those points cannot be called complete passes or replayed.
Final weights and both prescribed full-score checkpoints are retained. A future stopping rule must check all components and save the passing weights.</p></section>
<section><h2>Step 5 — Check representation and implementation separately</h2>
<p>For the one/eight-row controls, we kept the hidden layers at their initial weights.
We then solved the final linear layer directly by SVD least squares. This is like fitting coefficients to known answers,
without thousands of gradient updates. It uses the same model shape and the same input/output scales.</p>
{capacity}
<p>All four controls meet the budgets. The eight-row feature matrices have full row rank.
This is evidence that the model can fit these particular narrow sets. It is not evidence that the model predicts new states well.
The condition numbers warn that the solve can be sensitive; this is not a recommended general-purpose chemistry surrogate.</p>
<p>We also encoded, normalized and decoded the exact labels, without a learned model:</p>{roundtrip}
<p>The resulting errors are far below budget. Tests check that the physical-loss value and gradient match the direct budget-error formula.
Saved-model replay checks the final predictions and recomputes the scores. No scaling/reconstruction defect was found in these checks.</p></section>
<section><h2>Step 6 — Decide what to do next</h2>
<p><strong>Supported:</strong> training procedure and loss weighting matter. The current fixed-rate Adam setup is not accurate enough,
at its final checkpoint, even when the small model can represent the examples. Some accurate species fits are lost with continued updates.
There is no demonstrated approximation ceiling on the eight-row sets.</p>
<p><strong>Not supported:</strong> a useful chemistry surrogate, many correct relative digits for all tiny increments,
a robust winner among representations, or a reason to add residual stages now. Only one seed and one narrow parent were checked.</p>
<p><strong>Recommended next test:</strong> keep these same training-only sets and test a bounded optimizer-polishing stage
(for example, a decreasing step size or L-BFGS), with a predeclared all-component stopping rule and saved passing weights.
First establish repeatable training fit.
Then use additional, predeclared parent trajectories and seeds to test generalization. Do not run a broad search, full grid,
residual stack or full trajectory rollout yet.</p></section>
<section><h2>Reproduction and cost</h2>
<p>The 12 Adam fits and four head controls took {summary['elapsed_seconds']:.2f} seconds in a one-CPU, 4-GB, offline container on lh40902.
This is runner wall time, not a timing comparison between learning algorithms. The kernel did not support a separate swap limit.
Raw arrays, scales, predictions and weights remain under <code>runs/representation/fit-diagnostic-20261008</code> in the server project.</p>
<p><a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments/benchmarks/precision_conditioning/learning">Source and commands</a> ·
<a href="https://github.com/xiao312/DFODE-kit/pull/4">Draft PR #4</a></p><pre>{source}</pre></section></body></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    if summary["status"] != "complete" or len(summary["results"]) != 16:
        raise ValueError("A complete diagnostic is required")
    output = args.run / "review"
    print(json.dumps({"output": str(output)}))
    if not args.dry_run:
        output.mkdir(exist_ok=True)
        figures(summary, output)
        (output / "report.html").write_text(report(summary), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
