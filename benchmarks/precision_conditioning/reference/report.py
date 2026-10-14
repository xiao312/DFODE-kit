"""Build a static checkpoint report. Read saved evidence; do not run chemistry."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
from pathlib import Path


def table(headers, rows):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(summary, manifest, components):
    count = summary["assessed_species_components"]
    budget = summary["budget_fit_count"]
    relative = summary["relative_fit_count"]
    unresolved = [row for row in components if row["budget_fit"] == "False"]
    affected = len({row["record_id"] for row in unresolved})
    metrics = table(["Check", "Result", "Denominator"], [
        ["Budget-fit species increments", f"{budget:,} ({100 * budget / count:.2f}%)", f"{count:,} assessed components"],
        ["Relative-fit species increments", f"{relative:,} ({100 * relative / count:.2f}%)", f"{count:,} assessed components"],
        ["Unresolved budget check", count - budget, f"{affected} affected intervals"],
        ["Direct zero estimates", summary["reference_zero_estimate_count"], "Not a proof of exact physical zero"],
        ["Endpoint zero / direct nonzero", summary["endpoint_zero_direct_nonzero_count"], "Diagnostic count, not proof of physical accuracy"],
        ["Solver failures", len(summary["failures"]), sum(s["completed"] for s in summary["solvers"].values())],
    ])
    solver_rows = []
    for name, item in summary["solvers"].items():
        if name == "radau_reference":
            continue  # Its zero error against itself is not evidence of accuracy.
        error = item["species_budget_error_max_per_interval"]
        solver_rows.append([name, f"{error['p99']:.6g}", f"{error['max']:.6g}", f"{item['total_seconds']:.3f}"])
    solvers = table(["Solver / input setting", "p99 interval-maximum error / budget", "Maximum error / budget", "Solve seconds"], solver_rows)
    unresolved_table = table(["Interval", "Species", "Direct increment", "Uncertainty / budget"], [
        [row["record_id"], row["species"], f"{float(row['delta']):.7g}", f"{float(row['uncertainty_budget']):.5g}"]
        for row in sorted(unresolved, key=lambda row: float(row["uncertainty_budget"]), reverse=True)
    ])
    conservation = table(["Setting", "Mass change sum (absolute)", "Element change max", "Relative enthalpy drift", "Minimum Y"], [
        [name, *[f"{summary['solvers'][name][key]:.5g}" for key in
                  ("mass_delta_sum_abs_max", "element_delta_max", "enthalpy_relative_drift_max", "minimum_mass_fraction")]]
        for name in ("absolute21", "radau_reference")
    ])
    quantization = summary["solvers"]["fp32_input"]["input_quantization_budget_max_per_interval"]
    elapsed = manifest["elapsed_seconds"]
    extrapolated = elapsed * 2160 / summary["interval_count"] / 60
    provenance = {key: manifest[key] for key in ("started_utc", "status", "elapsed_seconds")}
    # Include numerical provenance, not package paths or arbitrary environment data.
    provenance["source"] = manifest.get("source", manifest.get("git", {}))
    provenance["versions"] = manifest.get("versions", {})
    provenance["mechanisms"] = [{key: item[key] for key in ("id", "sha256", "species_count", "reaction_count")}
                                for item in manifest["mechanisms"]]
    provenance_html = escape(json.dumps(provenance, indent=2))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkpoint 02 — Reference accuracy</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#182b3e;background:#f4f6fa;max-width:1150px;margin:auto;padding:28px}}
section{{background:white;padding:24px;margin:22px 0;border-radius:12px}}h1,h2{{line-height:1.2}}a{{color:#1d6094}}
img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #d6dfe8}}
.scroll{{overflow-x:auto}}.notice{{border-left:5px solid #b66b12;padding:15px;background:#fff4e3}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}
</style></head><body>
<nav><a href="../index.html">Research home</a> · <a href="https://github.com/xiao312/DFODE-kit/issues/3">Review in Issue #3</a> · <a href="summary.json">Download metrics</a> · <a href="analysis-provenance.json">Analysis provenance</a></nav>
<h1>Checkpoint 02: reference-accuracy feasibility</h1>
<p>Executed on lh40902. Four parent trajectories. {summary['interval_count']} intervals. {elapsed:.1f} seconds. No model training.</p>
<p class="notice"><strong>Recommendation: proceed with a filtered reference subset, after review.</strong>
Keep the {count - budget} unresolved components separate. Add independent parent trajectories before a train/validation/test split.
Do not generate the full grid or claim solver-level significant digits from this pilot.</p>
<section><h2>What passed?</h2>{metrics}
<p>A budget-fit component has estimated reference uncertainty at most 1% of
<code>1e-12 + 1e-6 * abs(Y_initial)</code>. Relative fitness also requires
<code>abs(delta) &gt; 100 * max(uncertainty, endpoint_spacing)</code>.
The relative check excludes {count - relative:,} of {count:,} components, including zero estimates.
These are empirical agreement checks, not rigorous error bounds.</p></section>
<section><h2>Solver agreement and label fitness</h2>
<img src="reference-feasibility.png" alt="Solver agreement across tolerance settings and label fitness by increment magnitude">
<p>The left plot and table use the p99 of the maximum species error in each interval, across 96 intervals.
This is not the pooled component p99. Errors use the direct Radau solution as a working reference, not exact truth.</p>{solvers}
<p>The Radau reference used {summary['solvers']['radau_reference']['total_seconds']:.1f} solve seconds.
Its error against itself is omitted. Tight relative tolerance alone did not remove the largest differences.
The initial-state budget is strict for species that form rapidly from a near-zero initial concentration.</p></section>
<section><h2>Where the reference check failed</h2>{unresolved_table}
<p>All unresolved species components occur in three H2 intervals at 1500 K, with duration 1e-4 s.
They include large product increments. Thus, reference disagreement is not only a trace-species problem.
Exclude these components, or exclude the three intervals as a conservative whole-sample rule.
Do not relax the physical budget after seeing these results.</p></section>
<section><h2>Cancellation and input precision</h2>
<img src="reference-cancellation.png" alt="Reference uncertainty against increment magnitude and endpoint-zero counts by species">
<p>The direct solver integrates changes from zero. It can retain changes that endpoint subtraction loses.
It still shares Cantera FP64 thermochemistry with the endpoint solver. The plotted very small changes
are numerical estimates; their magnitude is not evidence of many correct significant digits.</p>
<p>To isolate input effects, compare the same tight endpoint solver before and after rounding T, P and Y
to FP32, then restoring FP64 and normalizing Y. The p99 interval-maximum difference is
{quantization['p99']:.6g} physical budgets; the maximum is {quantization['max']:.6g}.
This changes the initial physical state. It is not an FP32 output-rounding test.</p></section>
<section><h2>Conservation and cost</h2>{conservation}
<p>Tiny negative mass fractions remain in the saved results. No clipping was used to hide them.</p>
<p>A linear estimate for 54 parent trajectories, eight anchors and five intervals is 2,160 intervals,
or about {extrapolated:.0f} minutes for all nine checks on this host. With 40 anchors it is about
{5 * extrapolated / 60:.1f} hours. This is a rough cost estimate. Other conditions and the unverified
Okafor mechanism can change the cost. It is not a request to start that run.</p></section>
<section><h2>Next experiment and limits</h2>
<p>The code has budget-linear, signed-power and scaled-asinh interfaces, stable inverse tests,
train-only normalization and a fixed FP32/FP64 comparison configuration. No models were trained.
The pilot has only two parents per mechanism. It cannot supply three independent data splits.
Add a small reviewed set of parents, then compare the three coordinates with matched model settings.</p>
<p>The direct method is SciPy Radau on scaled delta-T and delta-Y. Two settings and tight CVODES checks
estimate uncertainty. This is not arbitrary precision. Shared chemical-rate errors remain unmeasured.
The H2 and CH4 tests use Cantera h2o2.yaml and gri30.yaml. The Fuel-study Okafor NH3/CH4 files remain
pending source verification. No flame or full-trajectory surrogate result is claimed.</p></section>
<section><h2>Reproduction and provenance</h2>
<p>Raw data remain in <code>/data2/kexiao/work/projects/DFODE-kit/runs/reference-pilot/checkpoint02-20261008</code>.
The pilot container used one CPU, 4 GB memory, no network and a one-hour limit.</p>
<p><a href="https://github.com/xiao312/DFODE-kit/tree/research/precision-conditioned-increments/benchmarks/precision_conditioning/reference">Source, configuration and reproduction commands</a></p>
<pre>{provenance_html}</pre></section></body></html>'''


def build(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Publish only a complete pilot; inspect failures before publication.")
    with (output / "components.csv").open(newline="") as stream:
        components = list(csv.DictReader(stream))
    destination = output / "report.html"
    destination.write_text(build_html(summary, manifest, components), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"report": str(args.output / "report.html")}))
    if not args.dry_run:
        build(args.output)
