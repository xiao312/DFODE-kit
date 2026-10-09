# Fuel training-recipe baseline

## Responsibility and non-goals

Rebuild the two saved Fuel study training recipes on the checked offline data.
This first stage is a **reduced-data recipe replication**, not a reproduction
of the paper's dataset, figures, CFD results, or published accuracy. Original
assets and earlier experiments are read-only. No new CFD run is included.
The source audit is [here](../../../docs/agents/fuel-baseline-replication.md).

## Frozen first experiment

Use the existing 10,000 training and 1,023 development rows, separately for
seeds 20261011 and 20261012. Use the same seed-selected rows as the paired
campaign. Keep the primary increment rule and both error-scale grids unchanged.
Use 61 inputs, four 800-unit GELU layers, **58 outputs** (omit AR), FP32,
Adam and normalized L1 loss. Shuffle all training rows once per epoch; retain
the last partial batch for power; the state recipe drops it. Requested batch
size is 20,000, capped at training count. With 10,000 rows the actual batch is
10,000, and there is only one update per epoch. This cap is a declared change:
the original state script would execute zero batches on fewer than 20,000 rows.

| Recipe | Target normalization | Epochs | Learning rate |
| --- | --- | --- | --- |
| `fuel-state` | Center and sample standard deviation | 1500 | 1e-3 through epoch 501; 1e-4 through 1001; then 1e-5; reset Adam before epochs 502 and 1002 |
| `fuel-power` | Zero center and population standard deviation of signed tenth-root increments | 2000 | 1e-3, 1e-4, 1e-5, 1e-6 in four 500-epoch stages; retain Adam state |

State inputs use sample standard deviations; power inputs use population
standard deviations. Pressure uses a training-fitted scale, not the prior
fixed half-range. Species inputs use the algebraically affine-equivalent
`Y**0.1` feature to avoid cancellation before normalization. Keep all input
and target statistics train-only. A zero standard deviation becomes one and
is recorded. Reject invalid input states; do not copy historical abs/clipping.
Use stable FP64 reconstruction and report inverse-domain corrections. These
are explicit safety differences from the historical diagnostic code.

The paper used approximately eight million augmented states. The recovered
state training log used 7.6 million training rows, 570,000 updates and 11.4
billion row presentations. This first run uses 15 or 20 million presentations.
Equal epoch counts are **not** equal data exposure or a full replication.
Do not compare historical normalized losses as if the datasets were identical.
Dataset growth, interpolation/filter parity, and the paper's fixed temperature
switch are next factors; do not silently introduce them into this comparison.

## Interface, dependencies and security

`python -m benchmarks.offline_accuracy.paper_baseline.run <dataset> --audit
<audit> --base <original-training> --recipe fuel-power --seed 20261011
--output <new-directory> --dry-run` checks inputs without creating outputs.
Omit `--dry-run` to run. Existing outputs are refused. No network is used.
Use the existing project-local version-recorded GPU environment; no CPU training
fallback. Set `CUDA_VISIBLE_DEVICES` to an idle GPU and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each fit has a 1200-second limit. Run at most
one fit at a time. Source code, environment, data hashes and config are saved.
Raw states, original mechanisms, checkpoints and paths stay in ignored runs.

Dependency graph: `plan -> coordinates -> fit -> run`; `run` uses the existing
reference input gate, paired metrics/runtime and independent physical verifier.
No production code depends on this module. Saved results feed the existing
research review. Intermediate records are fixed-epoch diagnostics, not model
selection. The final checkpoint is always retained and replayed.

## Verification

`python -m pytest tests/test_paper_baseline.py -q`

Expected: source-specific normalization and stage boundaries pass; signed
increments round-trip; AR is not a network output; saved GPU weights replay.
CPU is permitted only for these small non-training checks and reference I/O.
Each live run must independently check acceptance counts and physical scores.
