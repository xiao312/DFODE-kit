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

## Separate 50k data-size stage

After the four 10k fits, prepare `dataset-50k.json` with the existing flame
preparer and run a fresh 16-state-per-split reference audit. This requires
Cantera/SciPy on one CPU; neural training remains GPU-only. Preparation has a
3600-second limit; audit has a 900-second limit. Fail closed on an incomplete
dataset, fewer than 50k accepted labels, or a failed audit.

Pass `--training-count 50000 --comparison-dataset <original-dataset>` to the
runner. It checks unchanged source/domain/chemistry, exact original training
prefix, exact development states AND labels, and preserves each seed's original
10k rows as the first 10k of its nested selection. No new development selection.
Train-only scales are refit; this is a full data-size protocol comparison, not
a fixed-scaler ablation. Keep both recipes and both seeds. State training drops
the final 10k partial batch per shuffled epoch, matching its source rule;
power training retains it. Record the different update/presentation counts.
This remains the existing four-source-snapshot domain, not eight-million-row
paper data or restored heat-release/interpolation lineage.

`python -m benchmarks.offline_accuracy.paper_baseline.campaign --source
<source-snapshots> --original <original-dataset> --base-root <original-campaign>
--output <new-campaign>` prints the six stages without mutation. `--execute`
runs them in order, saves separate logs and `campaign-status.json`, and stops
on any failure. The overall maximum is 3700 + 960 + 4*1500 seconds; actual
neural fits have their own shorter 1200-second limits. Completion requires all
four verified model results, not just a completed preparation process.

## Bounded 200k extension

The parallel reference runner prepares 200,500 candidates, reusing the complete
50k dataset's prefix and development labels. A fresh reference audit is required.
`--training-count 200000 --comparison-dataset <original-10k-dataset>
--nested-run <verified-50k-fit>` requires the same seed and recipe. It preserves
the saved 50k training row IDs as the first 50k selection. Missing or changed
provenance fails before GPU training. Both recipes and seeds remain required.
The frozen epoch schedules remain unchanged: 15,000 state-recipe updates and
20,000 power-recipe updates, with batch size 20,000. This comparison increases
data AND optimization exposure. It is not a matched-update data-only ablation.
Do not open the test set or claim a million-row result from this stage.

The existing campaign accepts `--training-count 200000 --previous-campaign
<complete-50k-campaign> --workers 8 --throughput-benchmark <verified.json>`.
It remains dry-run by default. Execution requires two exact-parity throughput
measurements for the selected worker count against that 50k dataset. Preparation,
audit and four sequential GPU fits retain the existing external time limits.

## Review output

The [matched-target experiment](matched_targets/README.md) reuses the fixed 200k
pool for a four-target, three-objective comparison under one common 18k schedule.
Its normalized-coordinate and physical losses remain separate named factors.
`evaluate_and_verify` accepts explicit preprocessing/prediction/reload callables
for this experiment; omitted callables preserve the original recipe evaluator.

The separate [matched-work experiment](matched_work/README.md) compares nested
50k and 200k pools with equal updates, batch sizes and row presentations. It
freezes 50k preprocessing and indexes the schedule by updates. Its results are
not mixed with the original fixed-epoch recipes.

`python -m benchmarks.offline_accuracy.paper_baseline.review <current-snapshot>
<complete-10k-campaign> --output <new-json>` requires all four verified fits,
matching dataset/audit identities, environment hashes and declared work counts.
It adds `fuel_models` and `fuel_history`; it preserves every existing query
and the report identity. The output has no raw chemistry states, weights,
credentials or machine paths. The existing report's `FuelBaseline.jsx` uses
the canonical recipe registry, both scoring rules and both retained seeds.
`python -m benchmarks.offline_accuracy.paper_baseline.scale_review <snapshot>
--campaign <complete-10k-campaign> --campaign <complete-50k-campaign>
--throughput <verified-benchmark.json> --output <new-json>` adds checked size
and CPU-throughput evidence. An optional third `--campaign` adds complete 200k
results only after all four fits pass. Prior queries and app identity stay intact.
