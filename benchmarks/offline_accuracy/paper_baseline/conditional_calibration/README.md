# Training-only conditional calibration diagnostic

## Responsibility and non-goals

Test whether scalar, output-dependent bias repeats on correction-unseen rows
within the authorized training pool. This is not a candidate qualification,
new loss sweep, temperature switch, or independent confirmation. Constant
calibration was weak; this check changes only conditional resolution.

## Inputs, outputs and method

`gradient_probe.run --experiment EXP-20261013-conditional-calibration` reuses
the verified 50k parent, admission, lease and Bubblewrap worker. Discovery is
the default. `--execute` runs two seeds, with one GPU and one CPU thread.
Config: two folds, seed+411 permutation, 16 quantile bins, minimum fit-bin
count128, s=1e-14. One fixed procedure per species and fold; no tuning.

Each fold fits a constant control and a 16-bin conditional map on the other
25k rows. Edges depend only on fitted base outputs in asinh coordinates.
Duplicate edges are removed, exact edges go to the right bin, and extreme
queries use the extreme bin. Sparse bins return zero correction. Both arms
use the existing interval-interior consensus rule and its physical zero
fallback. Conditional bins route by predicted increment, never a query label.
No claim of monotonicity, continuity or exact discrete optimum is made.

Outputs: fold identities, fitted edges/constants, check predictions, and
physical pass/repair/damage, sign, tail and endpoint counts. Cross-fit rows
were historically seen by the base; they are not independent test data.
Full U/D is not opened for this diagnostic. No deployable candidate is saved.

## Dependencies and security

`controller -> gradient_probe.run -> existing admission/sandbox/run_worker`;
`worker -> interval_calibration.train_constants -> checked asinh decoder`;
`review -> raw arrays + separately computed physical formula`. The controller
stages only allowed training labels and frozen base metadata. No secrets,
network, home, confirmation labels or production dependents. GPU closed-form
calculation is intentional; no CPU training fallback.

## Verification

Run `python -m pytest tests/test_conditional_calibration.py -q`. Expected:
duplicate/constant-bin handling, query-label-free replay, deterministic folds,
and known conditional shifts. Review must verify row exclusion, replay and
repair identity. Example discovery: `python -m benchmarks.offline_accuracy.
paper_baseline.gradient_probe.run --project <root> --output <new-run>
--experiment EXP-20261013-conditional-calibration` (join module text).

If conditional checks improve both seeds without severe damage, register a
full candidate comparison. Otherwise close scalar-bias calibration provisionally.
