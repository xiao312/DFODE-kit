# Checkpoint 01 — representation audit

Date: 2026-10-08. NumPy 2.5.3, Python 3.14.6, 1,169 synthetic pairs.
Defaults: diagnostic atol=1e-12, rtol=1e-6; transform exponent=0.1.

| Path | Nonzero targets lost | p99 relative error | Maximum budget error |
| --- | ---: | ---: | ---: |
| FP64 endpoint subtraction | 362 | 1.0000 | 1.0588e-10 |
| FP32 endpoints, FP64 subtraction | 638 | 1.0000 | 2.2452e2 |
| Naive Box-Cox endpoints/reconstruction | 0 | 1.8197e15 | 1.2143e-5 |
| Budget-linear, FP32 coordinate | 0 | 5.3843e-8 | 2.4001e2 |
| Budget-asinh, FP32 coordinate | 0 | 7.3455e-7 | 8.4482e3 |
| Signed-power, FP32 coordinate | 0 | 5.5818e-7 | 3.2579e3 |
| Budget-linear, FP64 coordinate | 0 | 1.4536e-16 | 1.3553e-8 |
| Budget-asinh, FP64 coordinate | 0 | 1.2194e-15 | 1.3878e-5 |
| Signed-power, FP64 coordinate | 0 | 5.1049e-15 | 1.7347e-6 |

All coordinate paths decode in FP64. This isolates target representation and does
not benchmark inference. The endpoint controls intentionally include naive arithmetic.
The grid covers only six initial concentrations and is not a chemistry distribution.

## Decision supported by these results

Audit the reference labels and cancellation before model comparisons. Coordinate
compression retains small targets but can amplify rounding during inversion. The
large relative error of naive Box-Cox reconstruction reflects tiny spurious increments;
its small budget error does not restore significant digits. Conversely, loss of a
tiny increment may be acceptable to a state budget while unacceptable to an increment
precision claim. Report both metrics and choose which goal the experiment tests.

No conclusion about neural learning benefit, speedup, chemical validity or novelty
follows from this audit.

## Verification and execution status

Seven focused numerical tests pass; syntax checks and `git diff --check` pass.
The full existing test suite could not collect because the lightweight environment
does not contain Torch or Cantera. This checkpoint is not ready for a merge based
on full-suite verification.

Execution was local. The PKU VPN reconnect returned `AuthenticationDidNotComplete`;
SSH to the development server timed out. No remote checkout or training job was created.
After connectivity returns, inspect compute resources and available schedulers before
selecting an experiment environment or submitting heavy work.
