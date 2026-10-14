"""Build an offline review page from the synthetic audit JSON."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def build_html(report):
    rows = []
    for result in report["results"]:
        rows.append(
            "<tr><td>" + escape(result["name"]) + "</td>"
            + f'<td>{result["nonzero_predicted_zero"]}</td>'
            + f'<td>{result["relative_error_p99"]:.3e}</td>'
            + f'<td>{result["budget_error_max"]:.3e}</td>'
            + f'<td>{result["fraction_within_budget"]:.2%}</td></tr>'
        )
    table = "\n".join(rows)
    metadata = escape(json.dumps({key: report[key] for key in ("versions", "budget", "scope")}, indent=2))
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Precision conditioning — first numerical audit</title>
<style>
body{{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#223}}
h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:10px;border-bottom:1px solid #ccd;text-align:left}}td:not(:first-child){{font-variant-numeric:tabular-nums}}
.scroll{{overflow-x:auto}}.note{{background:#eef3fa;padding:18px;border-radius:8px}}
pre{{white-space:pre-wrap;background:#f4f4f4;padding:16px}}code{{background:#f4f4f4}}
</style>
<h1>Precision conditioning: first review checkpoint</h1>
<p>Synthetic representation audit, {report['sample_count']} state/increment pairs.
No chemistry solver or neural network was evaluated.</p>
<div class="note"><strong>First finding:</strong> {report['state_addition_lost_nonzero']}
known nonzero increments disappear when added to their initial state in FP64.
Preserving a tiny increment separately and recovering it from stored endpoints are different tasks.</div>
<h2>Coordinate and storage comparisons</h2>
<p>Quantized coordinate experiments encode to the indicated dtype and decode in FP64.
Relative error is measured only for nonzero reference increments. Budget error uses
<code>atol + rtol × |initial state|</code>. These are diagnostic weights.</p>
<div class="scroll"><table><thead><tr><th>Path</th><th>Lost nonzero increments</th>
<th>99th percentile relative error</th><th>Maximum budget error</th><th>Within budget</th>
</tr></thead><tbody>{table}</tbody></table></div>
<h2>Interpretation for the next experiment</h2>
<p>All direct coordinate paths retain nonzero targets on this grid. Asinh is not automatically
the most accurate quantization scheme: it compresses large budget-normalized increments,
and its inverse amplifies coordinate rounding. Its possible learning advantage must be
tested on chemistry data under a controlled architecture and compute budget.</p>
<p>The naive Box–Cox endpoint path also introduces spurious tiny changes from reconstruction
rounding. Very large relative errors near zero can coexist with small state-budget errors.
Neither metric alone establishes scientific accuracy.</p>
<h2>Next review decision</h2>
<p>Select the chemistry reference mechanism, reactor constraints, timestep range and tolerance
ladder. A small H2 mechanism is the provisional smoke test; the selected NH3/CH4 mechanism
is the scientific target. Audit label uncertainty before training.</p>
<h2>Reproduction metadata</h2><pre>{metadata}</pre>
<p>Full per-magnitude results and synthetic inputs are in <a href="audit.json">audit.json</a>.
The table and explanation are embedded in this document and work offline.</p></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source.with_name("report.html")
    report = json.loads(args.source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(report), encoding="utf-8")
    print(json.dumps({"output": str(output)}))


if __name__ == "__main__":
    main()
