# Acceptance-directed method adaptation

## Responsibility and non-goals

Improve the fixed offline acceptance score by adapting existing approximation
methods. This is a bounded development campaign, not an untouched test, CFD
deployment, numerical certification, or an implementation of full ISAT/MSNN.
The existing data, mechanisms, tolerance rule and previous runs are read-only.

## Plan and selection

Keep the same 10,000 training rows per seed, 1,023 development rows and primary
budget `1e-15 + 0.1*abs(reference)`. Require all 58 non-AR species for state
acceptance. Keep the full tolerance grid, audited 16-state subset, negative
endpoints and conservation diagnostics. Do not select a different pass rule.

Start from each seed's verified 4,000-update transformed-state model. Run:

1. `continue-coordinate`: 4,000 more updates of its original coordinate loss.
2. `finetune-physical`: 4,000 updates of mean log1p physical normalized error.
3. `finetune-tail`: same, plus 0.25 times the mean worst-eight-species loss per
   state. This is a tail-loss adaptation, not the ADA-CVaR sampling algorithm.
4. `relative-correction`: a zero-output-initialized 4x256 correction, 4,000
   updates, in a base-relative asinh chart. At inference its scale is
   `1e-14 + abs(base_prediction)`, never a reference-dependent oracle scale.
5. `protected-correction`: same correction plus tail loss and a training-only
   non-regression penalty on components the frozen base already predicts within
   budget. Zero initialization preserves the base only before optimization.
6. `local-state` and `local-asinh`: cubic radial-basis interpolation using 128
   neighbors, degree-one polynomial and smoothing 1e-8. Standardize inputs using
   training data and retain their numerical-rank SVD basis. Predict conventional
   transformed-state or tolerance-scale asinh coordinates. This is a local
   approximation comparator, not certified tabulation or a neural-network win.

Use both original seeds, retain all trials, and evaluate fixed final checkpoints.
Rank by complete-state acceptance, then component acceptance, then measured cost.
Do not claim selected development scores are independent generalization evidence.
Any later adaptation must have a new named configuration and retained results.

## Interface, storage and security

`python -m benchmarks.offline_accuracy.improve.run <dataset> --audit <audit>
--base <original-seed-training> --refined <verified-long-state-model> --seed
20261011 --variant <name> --output <new-directory> --dry-run`

Omit `--dry-run` to run one member. Refuse existing destinations. The input base
and audit hashes must match. Outputs include configs, source/runtime identities,
weights or local tables, preprocessing, optimizer state, selected row identities,
learning curves, prediction arrays and small summary JSON. No raw state tables,
private mechanisms, model weights or machine paths enter Git or public reports.
Only ignored project-owned run directories are writable on the server.

At most two jobs concurrently, each half a CPU and 4 GiB, one numerical thread,
network disabled, existing immutable image. Each fit has a 900-second wall limit;
an outer one-hour timeout bounds the execution. No new packages or shared changes.

## Dependencies and dependents

`plan -> run -> neural/local -> coordinates`; `run/verify -> refinement.inputs,
saved long-model loader, offline metrics, independent physical verification`.
NumPy/SciPy implement local FP64 approximations; existing Torch implements neural
models. Shared evaluation saves and independently checks every prediction. The
existing GitHub review consumes small summaries. Production code does not depend
on this module. Sources and adaptations are recorded in
`docs/agents/acceptance-improvement-methods.md`.

## Verification

`python -m pytest tests/test_acceptance_improve*.py -q`

Expected: differentiable stable reconstruction matches NumPy; zero correction
preserves its base; non-regression/tail losses use only training truth; local
interpolation handles constant dimensions; replay and independent metric checks
pass. The live runner saves `verification.json` bound to `result.json` only after
exact saved-model replay, independent acceptance/physical checks and input hashes.
The check also reports base pass-to-fail and fail-to-pass transitions, so an average
gain cannot conceal damage to previously accurate components.
