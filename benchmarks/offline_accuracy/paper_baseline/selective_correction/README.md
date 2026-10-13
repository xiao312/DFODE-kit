# Learned selection of frozen corrections

## Responsibility and non-goals

Test whether an affine selector can repair more components than it damages.
Both GBCT and the historical10k SmoothL1 correction remain frozen. No new
correction, temperature switch, numerical integration, CFD or promotion.
This is a training-pool diagnostic, not an independent test.

## Inputs and fixed work

The registered manifest selects50k bound training rows outside the correction's
10k fitting rows and U20k. A seeded permutation assigns20k fitting,10k threshold
calibration and20k check states. The base and feature calibration saw some or
all rows historically. Record that exposure; no U/D labels enter this worker.

Two arms: species intercepts, or affine logits on119 existing features plus
own correction and absolute correction, scaled by fitting-only RMS. Both use
zero initialization,2,000 Adam updates, batch1,000 and fixed cosine1e-3→1e-4.
Only discordant pass-status components supply binary classification loss.
Same-status components remain in every physical score and tail calculation.

Calibration selects one global probability threshold from0,.01,...,1 and
always-base1.01. Maximize net passes subject to<=1% damage of base successes,
no nontrivial pass-count loss and no added nontrivial sign errors. Ties favor
fewer replacements, then larger threshold. This is empirical, not certified.
Both arms reject a correction whose endpoint is negative. Record this explicit
label-free constraint and raw-candidate results; do not clip predictions.
Final output is exact selection between two signed increments, not blending.

`EXP-20261013-selective-nonlinear` reuses the completed affine control and exact
row roles. It changes only the gate function class to one16-unit GELU layer
per species (114,202 parameters), with seeded hidden initialization and zero
output layer. Same inputs, objective,2,000updates and2M presentations; not equal
FLOPs or parameter count. Source control and every calibration rule stay fixed.
This is the second bounded selector trial, not an open architecture sweep.

## Boundary and dependencies

`gradient_probe.run -> selective_correction.prepare -> pinned training/candidate`
stages data; `worker -> correction_scaling` reconstructs frozen predictions,
then `model` fits the small gate. `review` independently replays logits and
selection, verifies split IDs and reconstructs physical scores. Candidate
prediction functions accept covariates/artifacts only. No references are used
by those functions. No production code depends on this module. No network,
host home, secrets or confirmation labels are mounted. One idleGPU/one thread.

Outputs include role IDs, gate weights, scales, calibrated thresholds, raw and
selected predictions, protected training-pool scores, runtime and hashes.
No end-to-end latency qualification is claimed for this diagnostic. A positive
result requires a separately registered state-only full-U/D run.

## Verification and example

`python -m pytest tests/test_selective_correction.py -q` checks exact branch
selection, calibration fallback/ties, repair versus damage denominators and
affine/intercept parameterization. After execution require independent recount,
frozen-prediction replay and exact split exclusion before interpretation.

Discovery: `python -m benchmarks.offline_accuracy.paper_baseline.gradient_probe.run
--project <root> --output <new-run> --experiment EXP-20261013-selective-correction`.
Add `--execute` only after discovery. No automatic model promotion.
