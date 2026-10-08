"""Assemble public research review pages; deployment is a separate Git operation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import shutil

from build_report import build_html


def landing_page(report):
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
<div class="badge">Checkpoint 01 · ready for scientific review</div>
<nav class="grid" aria-label="Research services">{navigation}</nav>
<section class="checkpoint"><h2>Latest result: where do tiny increments disappear?</h2>
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
<p>Select the chemistry mechanism, reactor constraints, timestep range and reference tolerance
ladder. A small H2 case is the proposed initial check; the selected NH3/CH4 mechanism is the
main scientific target. <a href="https://github.com/xiao312/DFODE-kit/issues/3">Review checkpoint 02</a>.</p></section>
<footer>Public research review · reproducible source on
<a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments">the research branch</a>.
<a href="publication.json">Publication metadata</a>.</footer></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    figure_path = args.source.with_name("coordinate-audit.png")
    if not args.source.is_file() or not figure_path.is_file():
        parser.error("audit.json and adjacent coordinate-audit.png must exist")
    if args.dry_run:
        print(json.dumps({"source": str(args.source), "figure": str(figure_path), "output": str(args.output)}))
        return
    report = json.loads(args.source.read_text(encoding="utf-8"))
    checkpoint = args.output / "checkpoint-01"
    checkpoint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source, checkpoint / "audit.json")
    shutil.copy2(figure_path, checkpoint / "coordinate-audit.png")
    (checkpoint / "report.html").write_text(build_html(report), encoding="utf-8")
    (args.output / "index.html").write_text(landing_page(report), encoding="utf-8")
    (args.output / ".nojekyll").touch()
    (args.output / "publication.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "precision_conditioning", "versions": report["versions"],
        "review_issue": "https://github.com/xiao312/DFODE-kit/issues/2",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": 6}))


if __name__ == "__main__":
    main()
