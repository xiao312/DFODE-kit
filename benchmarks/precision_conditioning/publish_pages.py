"""Assemble public research review pages; deployment is a separate Git operation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

from build_report import build_html


def landing_page(report, reference=None, learning=None, fit=None, polish=None):
    count = report["sample_count"]
    lost = report["state_addition_lost_nonzero"]
    cards = [
        ("Discuss direction", "Research questions, hypotheses and scientific choices", "https://github.com/xiao312/DFODE-kit/discussions/5"),
        ("Review checkpoint 01", "Results, figures and the decision on our next experiment", "https://github.com/xiao312/DFODE-kit/issues/2"),
        ("Read the Wiki", "Methods, vocabulary and how to reproduce experiments", "https://github.com/xiao312/DFODE-kit/wiki"),
        ("Track experiments", "Experiment status and review gates (account access required)", "https://github.com/users/xiao312/projects/1"),
        ("Review code", "Draft pull request with implementation and test evidence", "https://github.com/xiao312/DFODE-kit/pull/4"),
        ("Research hub", "Milestones, links and the experiment checklist", "https://github.com/xiao312/DFODE-kit/issues/1"),
    ]
    navigation = "".join(f'<a class="card" href="{url}"><strong>{escape(title)}</strong><span>{escape(description)}</span></a>' for title, description, url in cards)
    latest = ""
    if reference is not None:
        latest = f'''<section class="checkpoint"><h2>Checkpoint 02: chemistry reference accuracy</h2>
<p>{reference['interval_count']} intervals ran on lh40902. Of {reference['assessed_species_components']:,}
species increments, {reference['budget_fit_count']:,} passed the reference-agreement check.
{reference['relative_fit_count']:,} also passed the stricter relative-error mask. These checks are not rigorous error bounds.</p>
<img src="checkpoint-02/reference-feasibility.png" alt="Chemistry reference agreement and label fitness">
<p>This reference-only checkpoint did not include model training or full-grid generation.</p>
<p><a href="checkpoint-02/report.html">Open checkpoint 02</a> ·
<a href="https://github.com/xiao312/DFODE-kit/issues/3">Review the decision</a></p></section>'''
    status = "Checkpoint 02 · ready for scientific review" if reference else "Checkpoint 01 · ready for scientific review"
    next_step = ("Review the chemistry reference masks and the next small set of independent parent trajectories."
                 if reference else "Select the chemistry mechanism, reactor constraints, timestep range and reference tolerance ladder.")
    if learning is not None:
        latest = f'''<section class="checkpoint"><h2>Latest result: first model comparison</h2>
<p>{len(learning['variants'])} matched models compare budget-linear, signed-power and scaled-asinh targets
in FP32 and FP64. Chemistry and training ran on lh40902. This small test uses one held-out parent per mechanism.</p>
<img src="checkpoint-03/comparison.png" alt="Measured model errors against the zero-change baseline">
<p><a href="checkpoint-03/report.html">Read the step-by-step explanation and results</a> ·
<a href="checkpoint-03/summary.json">Download measured metrics</a></p></section>''' + latest
        status = "Checkpoint 03 · ready for scientific review"
        next_step = "Review the model errors and training curves before further data expansion or residual learning."
    if fit is not None:
        latest = '''<section class="checkpoint"><h2>Latest: can the model fit known answers?</h2>
<p>The narrow training sets can be represented by the model. Some one-example Adam runs briefly meet species budgets,
then lose that accuracy. No final Adam fit meets every budget. This is a training-only diagnosis, not a generalization result.</p>
<img src="checkpoint-03-fit/fit-errors.png" alt="Training-fit errors: direct head controls pass; Adam fits remain outside budget">
<p><a href="checkpoint-03-fit/report.html">Read the step-by-step fit diagnosis</a> ·
<a href="checkpoint-03-fit/summary.json">Measured results</a></p></section>''' + latest
        status = "Checkpoint 03 fit diagnosis · ready for scientific review"
        next_step = "Review a bounded optimizer-polishing test before expanding the data or adding residual models."
    if polish is not None:
        latest = '''<section class="checkpoint"><h2>Latest: repeatable fitting and data expansion</h2>
<p>All methods now save passing one-example fits across three seeds. Direct final-layer solving also passes the eight-example sets.
Neither iterative method passes those sets, and no method passes all existing training rows. The milestone is partial.</p>
<img src="checkpoint-03-polish/polish-results.png" alt="Multi-seed fit errors show narrow control success and unresolved larger-set fitting">
<p><a href="checkpoint-03-polish/report.html">Read the results and dataset expansion gates</a> ·
<a href="checkpoint-03-polish/summary.json">Measured results</a></p></section>''' + latest
        status = "Checkpoint 03 optimizer polishing · ready for scientific review"
        next_step = "Use the existing training rows to establish a scalable recipe, then test modest data growth on independent parents."
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DFODE-kit Research Review</title><style>
:root{{color-scheme:light;font-family:system-ui,sans-serif;color:#182b3e;background:#f4f6fa}}
body{{max-width:1100px;margin:auto;padding:40px 24px}}h1{{font-size:clamp(2rem,5vw,3.4rem);line-height:1.1;max-width:850px}}
p{{line-height:1.7;max-width:850px}}.eyebrow{{color:#466986;font-weight:700;letter-spacing:.08em;text-transform:uppercase}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin:32px 0}}
.card{{display:block;background:white;border:1px solid #d6dfe8;border-radius:12px;padding:20px;text-decoration:none;color:inherit}}
.card:hover{{border-color:#2774ad}}.card span{{display:block;color:#526476;margin-top:8px;line-height:1.5}}
.checkpoint{{background:white;border-radius:14px;padding:28px;margin-top:32px}}img{{width:100%;height:auto}}
.badge{{display:inline-block;background:#dfedfc;padding:5px 12px;border-radius:20px;font-size:.9rem}}
a{{color:#1d6094}}footer{{margin-top:36px;font-size:.85rem;color:#526476}}
</style></head><body>
<div class="eyebrow">DFODE-kit / Research workspace</div>
<h1>Precision-conditioned chemistry surrogates</h1>
<p>A shared place to review the evidence, discuss scientific choices and follow experiments.
The working direction is error-budget-aware coordinates, residual approximation,
stable reconstruction and controlled integration.</p>
<div class="badge">{status}</div>
<nav class="grid" aria-label="Research services">{navigation}</nav>
{latest}
<section class="checkpoint"><h2>Checkpoint 01: where do tiny increments disappear?</h2>
<p>Across {count:,} synthetic state/increment pairs, {lost} known nonzero increments disappear
when added to the initial state in FP64. Direct increment coordinates preserve them,
but coordinate compression can amplify inverse-rounding error.</p>
<img src="checkpoint-01/coordinate-audit.png" alt="Synthetic audit comparing lost increments and coordinate round-trip errors">
<p><strong>Scope:</strong> numerical representation only. No chemistry solver or model training
was evaluated. Asinh's possible learning benefit remains an experimental question.</p>
<p><a href="checkpoint-01/report.html">Open the complete report</a> ·
<a href="checkpoint-01/audit.json">Download metrics and synthetic inputs</a> ·
<a href="https://github.com/xiao312/DFODE-kit/issues/2">Discuss this result</a></p></section>
<section><h2>What needs a decision next?</h2>
<p>{next_step} The selected NH3/CH4 mechanism remains pending source verification.
<a href="https://github.com/xiao312/DFODE-kit/issues/3">Review checkpoint 02</a>.</p></section>
<footer>Public research review · reproducible source on
<a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments">the research branch</a>.
<a href="publication.json">Publication metadata</a>.</footer></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--learning-review", type=Path)
    parser.add_argument("--fit-review", type=Path)
    parser.add_argument("--polish-review", type=Path)
    args = parser.parse_args()
    figure_path = args.source.with_name("coordinate-audit.png")
    if not args.source.is_file() or not figure_path.is_file():
        parser.error("audit.json and adjacent coordinate-audit.png must exist")
    reference_files = ("report.html", "summary.json", "analysis-provenance.json", "reference-feasibility.png", "reference-cancellation.png")
    if args.reference_run and any(not (args.reference_run / name).is_file() for name in reference_files):
        parser.error("reference report, summary, provenance and both figures must exist")
    learning_files = ("report.html", "summary.json", "comparison.png", "learning-curves.png", "magnitude-errors.png")
    if args.learning_review and any(not (args.learning_review / name).is_file() for name in learning_files):
        parser.error("learning report, summary and three figures must exist")
    fit_files = ("report.html", "summary.json", "fit-errors.png", "fit-curves.png")
    if args.fit_review and any(not (args.fit_review / name).is_file() for name in fit_files):
        parser.error("fit report, summary and two figures must exist")
    polish_files = ("report.html", "summary.json", "polish-results.png")
    if args.polish_review and any(not (args.polish_review / name).is_file() for name in polish_files):
        parser.error("polish report, summary and figure must exist")
    if args.dry_run:
        print(json.dumps({"source": str(args.source), "figure": str(figure_path), "output": str(args.output)}))
        return
    report = json.loads(args.source.read_text(encoding="utf-8"))
    checkpoint = args.output / "checkpoint-01"
    checkpoint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, checkpoint / "audit.json")
    shutil.copy2(figure_path, checkpoint / "coordinate-audit.png")
    (checkpoint / "report.html").write_text(build_html(report), encoding="utf-8")
    reference = None
    if args.reference_run:
        reference = json.loads((args.reference_run / "summary.json").read_text())
        reference_destination = args.output / "checkpoint-02"
        reference_destination.mkdir(parents=True, exist_ok=True)
        for name in reference_files:
            shutil.copy2(args.reference_run / name, reference_destination / name)
    learning = None
    if args.learning_review:
        learning = json.loads((args.learning_review / "summary.json").read_text())
        learning_destination = args.output / "checkpoint-03"
        learning_destination.mkdir(parents=True, exist_ok=True)
        for name in learning_files:
            shutil.copy2(args.learning_review / name, learning_destination / name)
    fit = None
    if args.fit_review:
        fit = json.loads((args.fit_review / "summary.json").read_text())
        fit_destination = args.output / "checkpoint-03-fit"
        fit_destination.mkdir(parents=True, exist_ok=True)
        for name in fit_files:
            shutil.copy2(args.fit_review / name, fit_destination / name)
    polish = None
    if args.polish_review:
        polish = json.loads((args.polish_review / "summary.json").read_text())
        polish_destination = args.output / "checkpoint-03-polish"
        polish_destination.mkdir(parents=True, exist_ok=True)
        for name in polish_files:
            shutil.copy2(args.polish_review / name, polish_destination / name)
    (args.output / "index.html").write_text(landing_page(report, reference, learning, fit, polish), encoding="utf-8")
    (args.output / ".nojekyll").touch()
    (args.output / "publication.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "precision_conditioning", "versions": report["versions"],
        "review_issue": "https://github.com/xiao312/DFODE-kit/issues/3" if reference or learning or fit or polish else "https://github.com/xiao312/DFODE-kit/issues/2",
    }, indent=2) + "\n", encoding="utf-8")
    count = 6 + sum(len(files) for value, files in [(reference, reference_files), (learning, learning_files), (fit, fit_files), (polish, polish_files)] if value is not None)
    print(json.dumps({"output": str(args.output), "files": count}))


if __name__ == "__main__":
    main()
