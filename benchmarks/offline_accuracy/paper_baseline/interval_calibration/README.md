# Fixed, training-derived coordinate calibration

## Responsibility and non-goals

Test whether even a single constant per species is selected differently by
coordinate-error minimization and the fixed physical acceptance contract.
This is a closed-form diagnostic of objective alignment, not another neural
loss sweep, a reference lookup, temperature switch, or production calibration.

Both arms learn 58 constants from the same fixed 50k training rows. At inference,
apply the constant in the checked fixed-asinh correction decoder around the
continued GBCT base. No query reference, fitting library or temperature condition
is available to the worker. AR is unchanged. No output clipping is used.

## Algorithms and interface

`smooth_l1`: solve each species' constant Smooth L1 objective, beta=log(1.1),
by 64 bisection steps on its monotone derivative between min/max training target.
If the minimizer is flat, select the value nearest zero; return zero if optimal.
`interval`: form acceptable correction intervals from training increments and
the unchanged 1e-15 + 0.1*abs(d) allowance. Sort all interval endpoints. Count
coverage in each open gap and choose the midpoint with greatest coverage,
breaking ties by smallest absolute midpoint. Prefer zero if its physical pass
count is at least as large. A final physical recount can fall back to zero if
rounding makes the selected midpoint worse. Record all such fallbacks.

The interval rule intentionally omits isolated endpoint-only maxima. It is a
robust interior consensus rule, not a claim of exact discrete global optimality.
All calculations use FP64 on the project GPU. The stored constants are FP64.
Both algorithms have finite declared work and no evaluation-tuned parameters.
They have equal model size, not equal arithmetic work.

`python -m benchmarks.offline_accuracy.paper_baseline.local_function.run
--project <root> --output <new-dir> --experiment EXP-20261013-interval-calibration`
is a dry-run discovery path; `--execute` admits the registered experiment.
`worker train` receives training labels and
`worker infer` receives only state/ID arrays and frozen model artifacts. Outputs
include constants, training decisions, physical predictions, hashes and timings.

## Dependencies and security

Reuse research_contract admission, Bubblewrap and immutable scoring; reuse the
hidden_features worker controller and score_arm; reuse the existing base loader
and decoder. Production does not depend on this module. No secrets, raw data or
weights enter Git. Use one idle GPU; do not change shared environments.

## Verification

Run `python -m pytest tests/test_interval_calibration.py -q`. Expect closed-form
constant-loss checks, interval coverage and tie tests, zero fallback, and exact
decoder replay. Full candidates require all existing physical guards and cost
checks, even if the empirical training count is no worse than the base.
