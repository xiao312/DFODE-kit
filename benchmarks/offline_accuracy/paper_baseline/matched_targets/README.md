# Matched targets and error-scale losses

## Responsibility and non-goals

Separate target-coordinate and loss effects on the existing 200k training pool.
This is a controlled adaptation, not a Fuel or GBCT paper reproduction. Do not
generate chemistry data, open the independent test, select checkpoints by scores,
change tolerances, launch CFD, or modify historical results/shared environments.

## Frozen protocol

Run four targets (`state-boxcox`, `signed-power`, `scaled-asinh`, `gbct`) times
three objectives (`coordinate`, `increment`, `state`) times seeds 20261011/20261012.
All 24 fresh fits use the exact existing per-seed 200k rows, the same 1,023
development rows, 61 inputs, four 800-unit GELU layers, 58 non-AR outputs, FP32
learning, FP64 inverse/loss, batch 10k and 18k updates (180M presentations).
Use one explicitly selected idle RTX 4090, sequentially. No CPU fallback.

Fit every input and target mean/sample-standard-deviation on the same 50k
training prefix only. Freeze them thereafter. Input statistics must be identical
across all targets; target statistics must be identical across objectives.
Use signed tenth-root physical increments for power, stable Box-Cox changes with
lambda=.1 for state, asinh(delta/1e-14), and GBCT with a=.1, b=.5, h=1e-6 s.
The asinh scale is fixed a/r from the primary increment rule, not fitted to scores.
GBCT uses common standardization, not the released loader's target normalization.

Use Adam at 1e-3, 1e-4, 1e-5 in three 6k-update stages. Reset Adam at every stage
boundary for ALL targets and objectives. Use the same shuffled minibatches and
initial weights within a seed. The first 12k updates use normalized-coordinate L1.
The last 6k use either that same loss or mean log1p(abs(physical increment error)
/ allowance), with allowance 1e-15+.1*abs(delta) for increment and
1e-15+.1*abs(Y+delta) for state. Only training labels set loss allowances.
No label is needed at inference. Record and compare the exact 12k warmup weights.
No extra clipping, conservation projection, tail penalty or loss mixing is added.
Report the inherited Box-Cox inverse-domain correction and its possible zero
gradient; correction is not evidence of accurate chemistry.

The state-coordinate control is arithmetically identical to the prior 18k
transformed-state protocol. Require exact initial weights, preprocessing, warmup
where available, final weights and predictions against that saved control.
Power now uses centered/sample-deviation normalization and the common three-stage
schedule: it is NOT the old Fuel-power recipe. Historical controls stay separate.
All arms include full-pool train/development diagnostics every 1k updates and
retain the final checkpoint. Equal update counts are not equal wall times; FP64
physical loss adds cost. Report acceptance under both frozen grids, error tails,
whole-state passes, negative endpoints, corrections, SSPI and measured cost.
The 16-state reference check does not certify every label.

## Interface, configuration, outputs and security

`python -m benchmarks.offline_accuracy.paper_baseline.matched_targets.campaign
--campaign <completed-200k-campaign> --previous <completed-50k-campaign>
--original <original-10k-dataset> --base-root <original-paired-campaign>
--baseline <completed-18k-budget-campaign> --output <new-directory>` previews
the commands without writing. Add `--execute` to run. The single-fit `run` takes
the same paths plus `--target`, `--objective` and `--seed`; its default is also
read-only, but validates source data. Existing output directories are refused.
Set CUDA_VISIBLE_DEVICES and CUBLAS_WORKSPACE_CONFIG=:4096:8 before execution.
Use the existing project-local pinned Python environment, not a new install.

Per fit: 1200-second training limit, 1500-second outer limit. The finite queue
stops on failure and records it; no silent retries or alternate settings.
Use compute nodes/approved GPU workstations, never a shared HPC login node.
Save configs, source/package versions, hashes, weights, optimizer, preprocessing,
row IDs, warmup hash, histories, predictions and verification in ignored runs.
No secret, private mechanism, large numerical output or machine path enters Git.
The final campaign verifier checks all 24 results and paired identities, not
only child exit codes. No result may be advertised as complete before verification.

## Dependencies and verification

`campaign -> run -> fit/model/coordinates/plan`; `run -> matched_work.run.inputs`
reuses the immutable nested-data gate. `coordinates -> paper_baseline.standardize,
flame_conditioning.state_change, paired.GBCT, improve.state_inverse` reuses stable
arithmetic. `run -> paper_baseline.run.evaluate_and_verify` supplies explicit
coordinate/prediction/reload adapters to the existing independent metric and
physical checker. Defaults in that checker remain unchanged for historical code.
No production module depends on this experiment. GitHub review consumes only
verified small summaries, never private raw data or model credentials.

`python -m pytest tests/test_matched_targets.py tests/test_paper_baseline.py
tests/test_matched_work.py tests/test_budget_work.py tests/test_matched_review.py
-q -p no:cacheprovider`

Expected: fixed schedule and batches, transform round trips including zero,
NumPy/Torch inverse parity and finite gradients, correct physical loss scales,
GPU saved-model replay, fixed warmup and refusal of changed paired identities.
Torch-free local tests may skip GPU checks; run those on the server before launch.

## Baseline review

`python -m benchmarks.offline_accuracy.paper_baseline.matched_targets.review
<current-snapshot> <complete-campaign> --output <new-snapshot>` checks all 24 bound
result/verification/environment records and the common initialization/warmup
identities. It emits sanitized `matched_targets_*` queries for the existing HTML.
Remote saved-array checks remain identified as remote checks, not local replay.
All prior queries and the report ID stay unchanged. Both seeds and both error
scales remain visible. These development results are the frozen comparison
baseline, not independent-test evidence or a deployable chemistry solver.
