# Reference accuracy pilot

## Responsibility and non-goals

Check whether chemistry increments are fit for a controlled learning test. Do not
generate the full grid, train models, or claim exact labels. Use the existing branch,
issue #3 and draft PR #4. The pilot has four constant-pressure, adiabatic trajectories:
H2/air and CH4/air at 900 and 1500 K, 1 atm, equivalence ratio 1.

## Interface and files

`pilot.py --config pilot.json --output <new-run-directory> --dry-run` validates the
plan without integration or output writes. Omit `--dry-run` to run. Existing output
directories are refused. Each completed interval is appended to `intervals.jsonl`.
`manifest.json` records the plan, versions, mechanism hashes and source revision.
`trajectory-*.json` stores selected states. `summary.json` and CSV files hold metrics.
All generated files stay in ignored `runs/` storage.

The JSON config sets mechanisms, temperatures, physical durations, at most eight
anchors, chemistry intervals, tolerance pairs and a wall-time limit. `analysis.py`
computes metrics from raw intervals. `chemistry.py` owns the reactor and direct
increment integrations. `targets.py` provides three reversible target coordinates
and train-only standardization. `learning.json` prepares a fixed comparison; no
training starts from the pilot command.

## Numerical method

Use Cantera 3.2.0 ReactorNet (CVODES) for endpoints. Start all interval solves from
the same saved FP64 state. Test `(rtol,atol)` pairs `(1e-9,1e-15)`, `(1e-10,1e-15)`,
`(1e-12,1e-15)`, `(1e-12,1e-18)`, and `(1e-12,1e-21)`. Repeat the last pair with
maximum step `interval/10`. Do not automatically run the optional `1e-24` stress test.

An independent check uses SciPy Radau to integrate delta-T and delta-Y from zero.
The RHS uses Cantera ideal-gas rates at the initial state plus the accumulated change.
It does not derive the reported increment by subtracting endpoint states. The change
variables use fixed physical-budget scales. Two Radau tolerances check convergence.
This changes the solver and state formulation but shares FP64 thermochemistry. It
is not arbitrary precision and does not establish accuracy below rate-evaluation
roundoff. No clipping or mass-fraction normalization is used in this RHS.

Eight anchors contain the initial state, logarithmic time samples, and points around
the greatest temperature rise in a 129-point scout trajectory. These anchors are
not a uniform chemistry sample. Save their times and temperatures. Non-ignition and
failed solves remain in the evidence. Initial states rounded to FP32 are re-integrated
with the same tight endpoint settings; record normalization by the reactor separately
from raw input quantization. Keep each trajectory as a group for later splits.

Estimate reference uncertainty using tight CVODES, step-limited CVODES, and the two
direct Radau solutions. A species component is budget-fit only when this estimate
is at most 1% of `1e-12 + 1e-6*abs(Y0)` and the checks are available. Increment-relative
fitness additionally requires a nonzero increment greater than 100 times the larger
of the uncertainty and endpoint spacing. Separate zero estimates, unresolved
components, and failed checks. Neither test is a rigorous error bound.

## Dependencies and security

Use Python 3.10 through 3.13 and `requirements.txt`. Cantera, NumPy and SciPy are required
for chemistry; Matplotlib is used only for exported figures; pytest runs tests.
The module depends on no production DFODE-kit training code. The existing Pages
publisher and checkpoint issue consume the selected report and figures.
There are no credentials, network calls or remote commands in the runner.

## Minimal example and verification

```bash
python -m pip install -r benchmarks/precision_conditioning/reference/requirements.txt
python benchmarks/precision_conditioning/reference/pilot.py --dry-run
python -m pytest tests/test_reference_pilot.py tests/test_precision_targets.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 3600 \
  python benchmarks/precision_conditioning/reference/pilot.py --output runs/reference-pilot/run-001
```

Expected: four trajectories and up to 96 interval records, with explicit failure
records if a solver fails. A one-hour external timeout is required for live runs.
Use one CPU worker. In Docker, also set `--cpus 1 --memory 4g`. Inspect metadata,
counts, conservation, and solver agreement before publishing. Fast tests must check
the real direct-increment RHS, target inversion, split isolation and metric masks.

`figures.py <run-directory>` exports two PNG files for inline GitHub review. It reads
only completed run metrics and raw intervals. `report.py <run-directory>` creates
the checkpoint HTML using the same static report format as checkpoint 01. It does
not publish. Use `--dry-run` on both commands to inspect their destinations first.
Keep the raw run on the server. Publish only the report, summary and figures.
`analysis-provenance.json` records the raw-input hashes, pilot source revision and
analysis revision separately. Publish this small file with the report. Re-analysis
does not alter the original manifest or claim that the pilot used newer source.
