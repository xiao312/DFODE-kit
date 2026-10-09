# Offline accuracy refinement

## Responsibility and non-goals

Separate additional optimization, tolerance-derived coordinates, physical loss,
and a frozen-base residual correction. This is a bounded development experiment,
not CFD validation, a solver replacement, or a reproduction of MSNN.

## Frozen comparison

Use the existing audited pressure-augmented dataset, the same nested 10,000
accepted training rows, and seeds 20261011 and 20261012. Keep the primary budget
`abs(error) <= 1e-15 + 0.1*abs(reference)` and the full existing tolerance grid.
The repeatedly inspected validation snapshots are development data, not an
untouched test. Do not increase the dataset or change the pass rule in this run.

For each seed, run these ten fixed-final-checkpoint models:

1. Four original coordinates, 4x800 GELU, 4,000 updates from the same original
   initialization. Compare with the saved 2,000-update models. This is a fresh
   longer cosine schedule, not continuation of an unavailable Adam state.
2. Budget signed-log and budget asinh, 4x800, 2,000 updates each. Compare with
   the original 2,000-update models. Use a common fixed coordinate scale of 100,
   zero offset, and no species-wise output standardization. Their derivative
   allocates numerical error; learned accuracy is still an experimental result.
3. Repeat both new targets with an added physical loss at the same 2,000 updates.
   Loss = coordinate L1 + 0.01 * mean(log1p(abs(error)/budget)). The robust log
   term prevents the initial largest errors from dominating every gradient.
   It is a surrogate objective, not an exact pass-rate optimizer.
4. One 4x800 residual model, 2,000 updates, on the frozen original 2,000-update
   transformed-state base. Train physical residuals in FP64, scaled by their
   per-species training RMS (floor 1e-30). Freeze the base and add its prediction
   and the correction in FP64. No validation labels enter residual construction.
5. An 8x800 single-network transformed-state control, 2,000 updates. Compare
   with the two 4x800 networks and the 4,000-update conventional model. These
   are approximate parameter/work controls, not equal measured CPU time.

All networks use FP32, Adam, batch 256, cosine learning rate 1e-3 to 1e-5,
identical sampled batches within a seed, and final checkpoints. Record actual
parameter count, CPU/wall training time, and repeated full-batch inference cost.
The residual result includes base training and inference cost. No seed selection.
Failures remain visible; never silently replace failed variants.

## Interface and configuration

`python -m benchmarks.offline_accuracy.refinement.run <dataset> --audit <audit>
--base <original-seed-training> --seed 20261011 --output <new-output> --dry-run`
validates data, base provenance, planned models and new destination. Omit
`--dry-run` and select `--variant <name>` to run one bounded model. `plan.py`
owns the immutable matrix. Each model has a 900-second internal wall limit;
use a one-hour external timeout and at most two half-CPU/4-GiB jobs in the
existing image. Inputs and solver installations are read-only.

Outputs: configuration/provenance JSON, training-only preprocessing, weights,
optimizer state, exact selected row identities, predictions, learning curves,
tolerance/SSPI/species/magnitude metrics, audit-subset uncertainty scores and
physical conservation checks. `verify` reloads weights, independently checks
acceptance counts, rebuilds training-only scales and checks frozen-base hashes.
It also evaluates `base + (reference - base)` in FP64 before any learned residual.
This exposes digits lost by residual formation/reconstruction; FP64 alone does
not guarantee all small increments survive. Refuse existing output paths.
No credentials, original mechanism, state arrays or weights enter Git or Pages.

## Dependencies and dependents

`plan -> run -> fit -> coordinates`; `run/verify -> flame data, model factories,
stable coordinates and physical metrics`; `run -> offline metrics/audit subset`.
NumPy implements target round trips; PyTorch implements differentiable inverses.
The existing report consumes only small verified summaries. Production code and
historical benchmark paths do not depend on this module.

## Verification

`python -m pytest tests/test_offline_refinement*.py -q`

Expected: signed/zero/extreme-scale round trips, finite zero derivatives,
NumPy/Torch inverse agreement, finite-difference gradients, fixed comparison
matrix and failure validation. The pinned server image supplies Torch. A small
synthetic fit checks save/reload and residual-base preservation before live runs.

`review <existing-snapshot> <results-root> --output <new-json>` requires all 20
hash-bound verified results and appends `refinement_*` queries. It refuses mixed
dataset identities and leaves earlier evidence unchanged. Report selection is
tested with `node --test tests/refinement_report_selection.test.mjs`.
