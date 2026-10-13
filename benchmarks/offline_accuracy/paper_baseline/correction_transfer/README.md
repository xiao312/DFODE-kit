# Frozen correction transfer and targeted reference audit

## Responsibility and non-goals

Diagnose the completed coordinate Smooth L1 heads without fitting anything.
Separate correction fitting, same-source transfer and held-out snapshot transfer.
Check whether the recorded sign-reversal tail has trustworthy reference labels.
Do not update labels, checkpoints, feature scaling or the acceptance rule. No
independent test, CFD, new training, model selection or automatic size progression.

## Fixed protocol (2026-10-13)

Read the immutable correction_scaling caches and the two completed Smooth L1
results. Check their hashes and recipe before use. Evaluate three populations:
the original 10k head-training states; 20k correction-unseen states (5k per training
snapshot, uniform without replacement from positions >=10k, RNG 20261013); and
all 1,023 development states. Selection never uses correction errors. Save exact
indices and count overlap with the 50k feature-calibration pool. The middle
population is base-seen, not an independent test of the full predictor.

Use gamma = 0, 0.125, 0.25, 0.5, 0.75, 1 in the fixed asinh decoder. This is
an evaluation curve, not a development-selected policy. Physical acceptance is
always `abs(error)<=1e-15+0.1*abs(reference)`. Record repairs/base failures and
damage/base successes, error tails, species/magnitude groups and snapshots.
For gamma=1, stratify base successes by initial normalized error in [0,.1],
(.1,.5],(.5,.8],(.8,1]. Record coordinate interval violations and whether the
correction moves away from the reference center or overshoots it. Check interval
decisions against physical decoding and report roundoff boundary disagreements.

Targeted audit: select 64 distinct tail-containing training states across both
seeds, 16 per snapshot. Cycle species/magnitude strata using a fixed deterministic
shuffle (20261013), not recomputed references. Each selected state has one anchor
component. Add 16 distinct same-sign controls: one per four tail anchors, matched
on seed, species, magnitude bin and snapshot, then nearest temperature. Exclude
all top-tail-containing states of either seed from controls. Fail if support is
insufficient; do not silently broaden the matching rule. Save all matching gaps.
The plan and exact states are frozen and hashed before any new integration.

Recompute fixed-T/V chemistry over 1e-6 s: CVODE at (rtol,atol)=(1e-12,1e-18),
(1e-12,1e-21), and the latter with max step=1e-7; independently integrate scaled
increments with SciPy Radau at (1e-9,1e-5) and (1e-11,1e-7). Radau absolute
tolerances apply to scaled increments, not physical mass fractions. Share the
existing mechanism/rate implementation, not CVODE's endpoint formulation.
Use direct-tighter as reference; uncertainty is maximum disagreement among all
five checks plus endpoint FP64 spacing. Qualify a component if uncertainty is
<=0.1 physical allowance. Report nominal-label differences in allowance units,
sign consistency, and selected-anchor results separately from all species.
Agreement is empirical evidence, not rigorous certification or FP64-digit proof.

CPU is required for these established Cantera/SciPy reference solvers. Bound the
audit to four single-thread workers, 120 seconds per state and 1,200 seconds
overall. Save failures and partial evidence; do not silently replace a failed
reference. GPU is used for frozen neural inference, with no CPU fallback.

## Interfaces and security

`python -m benchmarks.offline_accuracy.paper_baseline.correction_transfer.prepare
--original <correction-scaling-root> --candidate <sl1-root>
--dataset <original-200k-dataset> --output <new-plan-directory>` validates only.
Add `--execute` to freeze selections. Inputs remain read-only; outputs must be new.

`...correction_transfer.evaluate --plan <plan-directory> --original <root>
--candidate <sl1-root> --output <new-evaluation-directory> [--execute]`

`...correction_transfer.audit --plan <plan-directory> --dataset <200k-dataset>
--output <new-audit-directory> [--execute]`

No credentials or private mechanism contents enter Git. Large numerical inputs,
predictions and raw solver records remain under ignored project-local `runs/`.
Reports bind source revisions, environment versions, inputs, selected rows and
output hashes. Fail on mismatched provenance, output collision or nonfinite data.

## Dependencies and verification

`prepare -> selection / correction_scaling checked readers`;
`evaluate -> metrics / correction_scaling model and decoder`;
`audit -> chemistry / frozen selection`. Historical modules and production code
do not depend on this diagnostic. GitHub evidence is a downstream consumer only.

Run `python -m pytest tests/test_correction_transfer.py -q -p no:cacheprovider`.
Expected checks: deterministic disjoint selections, matching strata, unchanged
gamma-zero predictions, exact interval/physical agreement away from roundoff,
damage denominators and reference-quality qualification. Replay known 10k and
development predictions exactly before reporting new transfer results.
