# Low-step base continuation

## Responsibility and non-goals

Test whether a lower, fixed training schedule and larger gradient batches can
improve the continued GBCT predictor. This is a complete trained-model test,
not an asinh correction or an extrapolation from single-step diagnostics.
No new physics, labels, confirmation data, routing or output projection.

## Interface and configuration

Use the existing `local_function.run` controller with registered experiment
`EXP-20261013-base-polish`. Discovery is the default; `--execute` runs four fits.
`small` uses 1,000 rows/update; `averaged` uses 10,000, accumulated in 1,000-row
microbatches. Both use the same 50k pool, bases, preprocessing, 3,000 updates,
fresh Adam, FP32 parameters, FP64 decoding/loss, cosine 1e-7 to 1e-8 and final
checkpoint. These are matched updates, not matched presentations or elapsed time.
The sole primary factor is gradient batch size. No checkpoint is chosen on U/D.

Inputs: pinned parent review/data, states, training increments and frozen model.
Outputs: optimizer/weights, training history, exact replay, full T50k/U20k/D1023
predictions, protected score/gates, independent recount and measured latency.
The common prediction schema has an empty `coordinate` array because no
correction head exists. It is not an inferred zero correction.

## Dependencies and security

`prepare -> verified local_function parent`; `worker -> adaptive.fit` with fixed
preprocessing and no development diagnostics; `inference -> correction_scaling`
fixed-batch base loader/predictor. `local_function.run/review/postcheck` owns
admission, Bubblewrap, GPU contention, scoring and verification. Training sees
only the approved pool; inference sees states and frozen artifacts only. The
anchor copy is used for timing/recount, not candidate prediction. No network,
secrets, host home or confirmation access. One idle GPU and one CPU thread.

## Verification

Run `python -m pytest tests/test_base_polish.py -q`. Check microbatch gradients
against a full-batch mean, including unequal chunks. Require unchanged initial
base predictions, frozen preprocessing, final optimizer work, exact state-only
saved-model replay, independent scores and end-to-end latency before conclusions.
Example: `python -m benchmarks.offline_accuracy.paper_baseline.local_function.run
--project <root> --output <new-run> --experiment EXP-20261013-base-polish`.
