# Paired error scales and GBCT target

## Responsibility and non-goals

Test the GBCT target adaptation and compare increment-scaled with state-scaled
training. This is a fixed offline development experiment, not a paper reproduction,
new test set, CFD test or solver certificate. Historical runs are read-only.

## Frozen plan

Use the existing checked 10,000 training rows per seed and 1,023 development rows.
Keep seeds 20261011/20261012, four 800-unit GELU layers, FP32 learning and FP64
reconstruction. Compare `state-boxcox` and `gbct` targets. GBCT uses a=0.1,
b=0.5 and h=1e-6 s. Both use the same train-only mean/RMS normalization, not
the released GBCT loader's mean-absolute-label normalization. See the
[source check](../../../../docs/agents/gbct-source-and-adaptation.md).

For each target, run three objectives: `coordinate`, `increment`, `state`.
All fits start from identical seeded weights and use the same batches. Run 2,000
coordinate warmup updates, then 2,000 updates of the selected objective. Reset
Adam for phase two for all three arms. Warmup learning rate is 1e-3 to 1e-5;
phase-two rate is 1e-4 to 1e-6. Both schedules are cosine. Physical loss is mean
log1p(abs(error)/budget); it does not use development labels. Keep the fixed final
checkpoint. This is 12 fits, with no result-based selection or hyperparameter search.

Both physical objectives use a=1e-15 and r=0.1. Increment loss uses abs(delta);
state loss uses abs(Y+delta). Reference labels determine training loss only; the
decoder needs no reference. Evaluate each prediction under both named policies
and the old 3x4 tolerance grid. Also report a separately named application-state
pair (a=1e-12,r=1e-6). That pair is diagnostic, not a training objective.
Keep zero controls, error tails, species/magnitude bins, all-58-species acceptance,
physical checks and separately qualified 16-state reference evidence. No claim
about all labels follows from that subset. Record inverse-domain corrections.

## Interface and security

`python -m benchmarks.offline_accuracy.paired.run <dataset> --audit <audit>
--base <original-seed-training> --seed 20261011 --variant gbct-coordinate
--output <new-directory> --dry-run`

Omit dry-run to execute one fit. Existing outputs are refused. Input hashes,
configuration, weights, preprocessing, predictions, cost and replay verification
are saved only under ignored project runs. No secrets or raw mechanisms enter Git.
No network or shared environment writes. Use the existing immutable research
image, at most two half-CPU containers, 4 GiB each, one numerical thread, 900 s
fit wall limit and one-hour campaign limit. Failure remains visible.

## Dependencies and dependents

`plan -> run -> fit -> coordinates`; the read-only input gate is
`refinement.run.inputs`. Stable state transforms, model construction and physical
diagnostics come from flame_conditioning. `verify` independently counts both
policies and replays weights; downstream GitHub review consumes small summaries.
Production code does not depend on this module. Dataset growth is a later factor.

## Verification

`python -m pytest tests/test_paired_accuracy.py -q`

Expected: stable GBCT round trips, explicit inverse-domain correction, finite
Torch gradients, and scale-dependent acceptance with an identical error numerator.
The live runner verifies saved predictions exactly and checks both budgets,
training-only preprocessing, hashes, counts and physical diagnostics before
writing `verification.json`. Dry-run makes no output directory.
