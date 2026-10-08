# Precision conditioning experiments

## Responsibility

Measure representational error before training chemistry surrogates. This benchmark
uses explicitly known synthetic increments to separate endpoint subtraction,
coordinate encoding, and state reconstruction. It does not measure neural-network
accuracy or establish a chemical reference solution.

## Checkpoint 02 reference pilot

The `reference/` submodule checks H2 and CH4 chemistry label fitness. It is separate
from the synthetic audit and production labeling. See `reference/README.md` for
the bounded run, independent increment check, and target comparison interface.

## Inputs and outputs

The default deterministic grid contains nonnegative initial mass fractions and signed
increments spanning 1e-32 through 1e-2. Invalid negative endpoints are excluded.
`--atol` and `--rtol` define a diagnostic error budget
`w = atol + rtol * abs(y0)`; these are evaluation weights, not solver tolerances or
a guarantee on global integration error.

Output is JSON with reference inputs, error distributions, magnitude bins, dtype,
and Python/NumPy versions. Results go under the ignored `runs/` directory. A dry run
prints the planned grid size and destination without writing anything.

## Dependencies and boundary

Python and NumPy only. No solver, model, network, credentials, or GPU access.
This is independent of production preprocessing; later training experiments consume
the same metrics but must retain their real-data lineage and label tolerances.

## Example and verification

```bash
python benchmarks/precision_conditioning/audit.py --dry-run
python benchmarks/precision_conditioning/audit.py --output runs/precision-conditioning/audit.json
python benchmarks/precision_conditioning/build_report.py runs/precision-conditioning/audit.json
python benchmarks/precision_conditioning/plot.py runs/precision-conditioning/audit.json
python -m pytest tests/test_precision_conditioning_audit.py -q
```

Expected: finite metric summaries, signed targets and exact zeros retained, and
separate counts for increments lost in endpoint subtraction and state addition.
Even FP64 cannot recover an increment that was lost when its endpoint was stored.

The signed-power baseline is `sign(d) * abs(d)**0.1 / 0.1`.
The asinh baseline is `asinh(d/w)`. Both are encoded to FP32 or FP64 and decoded
in FP64. This isolates target quantization; it is not an end-to-end FP32 pipeline.
The transformed-state baseline uses naive endpoint Box-Cox subtraction intentionally
to quantify that failure mode. It is not a stable implementation recommendation.

Human review and decisions live in the
[research hub](https://github.com/xiao312/DFODE-kit/issues/1) and
[checkpoint 01](https://github.com/xiao312/DFODE-kit/issues/2).
The optional `plot.py` renderer depends on Matplotlib and produces a PNG suitable
for inline GitHub attachments. The numerical audit itself requires only NumPy.

`publish_pages.py` builds the public review site into an explicitly selected output
directory. It consumes the audit JSON and its adjacent `coordinate-audit.png`, uses
the sibling HTML renderer, and writes a landing page, checkpoint report, metadata and
downloadable audit. It performs no network calls or Git operations. Use `--dry-run`
to inspect its inputs and output first. Publish only these generated scientific
artifacts; credentials and server logs are outside this interface.

Pass `--reference-run <completed-run-directory>` to include checkpoint 02. This
copies only its HTML, two figures, summary and analysis provenance. Raw intervals,
environment details and large arrays are not public artifacts. Generate the report
with `reference/report.py` first. The landing page then links both checkpoints.
Pass `--learning-review <run>/review` to add checkpoint 03. Its allowlist contains
HTML, three exported PNG figures and measured summary JSON, not weights or arrays.
Pass `--fit-review <fit-run>/review` to include the checkpoint 03 training-fit
follow-up. Its allowlist is HTML, summary JSON and two figures. It is not checkpoint
04 (residual learning). The fit report depends only on saved diagnostic results.

```bash
python benchmarks/precision_conditioning/publish_pages.py runs/precision-conditioning/audit.json --output runs/research-pages --dry-run
python benchmarks/precision_conditioning/publish_pages.py runs/precision-conditioning/audit.json --output runs/research-pages
python -m pytest tests/test_precision_conditioning_pages.py -q
```

Review the [published report](https://xiao312.github.io/DFODE-kit/),
[Wiki](https://github.com/xiao312/DFODE-kit/wiki),
[research Discussion](https://github.com/xiao312/DFODE-kit/discussions/5), and
[Project](https://github.com/users/xiao312/projects/1) (account access required).
The site is deployed from `research-pages`, separately from the research source.
Before changing the existing documentation Pages workflow, reconcile its deployment
with this site; GitHub Pages serves one site for this repository.
