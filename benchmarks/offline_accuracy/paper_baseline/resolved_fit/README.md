# Small resolved training-set fit diagnostic

## Responsibility and non-goals

Test whether the existing correction features and head can fit a small, checked
set. This is a training-fit test, not a generalization result. Keep the original
GBCT base frozen. Do not expand data, tune on development, score the independent
test, change the pass rule, or replace any historical experiment.

## Fixed protocol

Use all 16 `train` records in the existing 200k reference audit, sorted by source
row. Require every row to be present in both seed-specific training pools. Do
not select rows using new predictions. Use the audit reference increments as
targets in every arm, not the ordinary endpoint labels. For each non-argon
component, require empirical reference uncertainty <= 0.1*(1e-15+0.1*abs(d)).
Include endpoint FP64 spacing in uncertainty. Exclude unresolved components from
both objectives and primary scores. Report their count. This screen is not a
rigorous reference-error bound.

Eight fits: seeds 20261011/20261012, 3x256/3x512 GELU correction heads, and two
objectives. Keep the fixed-crossover asinh decoder, s=1e-14, and the original
119 input features. Freeze preprocessing on the same 50k training prefix. The
output coordinate has offset zero and divisor one. All heads start at exactly
zero correction. The objectives, on the same resolved components, are:

- `direct`: mean square error against c*=asinh(d/s)-asinh(base/s).
- `physical`: mean log1p(abs(prediction-d)/(1e-15+0.1*abs(d))).

Use fresh Adam, 6000 full-batch updates, cosine learning rate 1e-4 to 1e-6,
FP32 heads, FP64 target/loss/reconstruction, no clipping, final checkpoint.
Report curves every 500 updates, raw gradient norms, parameters and GPU cost.
Widths have equal updates, not equal compute. Cap each fit at 600 seconds.
No early success stop or best-checkpoint selection. Before fitting, record FP32
oracle reconstruction and exact FP32-feature duplicate groups. Duplicate features
alone are not proof of incompatible physical predictions: the decoder also
depends on the base. No development or test metrics enter this diagnostic.

Scores: resolved component acceptance, resolved nontrivial acceptance
(abs(d)>1e-15/0.9), failures on components the base failed, sign errors,
per-species acceptance, p99/max normalized error and accepted resolved-subset
rows. Call the latter whole-state acceptance only if every active component is
resolved. Predeclare fit feasibility as >=99% resolved acceptance AND >=99%
resolved-nontrivial acceptance in both seeds. This threshold is a diagnostic
milestone, not a change to the component pass rule or a deployment claim.

## Interface and security boundary

`python -m benchmarks.offline_accuracy.paper_baseline.resolved_fit.run
--campaign <200k-campaign> --previous <50k-campaign> --original <original-data>
--base-root <original-paired-root> --baseline <matched-target-campaign>
--output <new-directory>` validates inputs without writes. Add `--execute` to
run the bounded eight fits on an idle GPU. Use the pinned project environment
and CUDA_VISIBLE_DEVICES with CUBLAS_WORKSPACE_CONFIG=:4096:8. No CPU fallback.
An external timeout should bound the complete command to 6000 seconds.

Inputs are checksummed datasets, audit and pinned baseline artifacts. Outputs
are a plan, environment, sample/mask arrays, frozen preprocessing/base weights,
head checkpoints, histories, predictions, metrics and result-bound verification.
Keep arrays/checkpoints in ignored runs. No credentials or network access are
needed. Output must be new; fail on missing rows, changed hashes, nonfinite
values, non-idle GPU or replay disagreement. Do not modify shared environments.

## Dependencies and verification

`run -> data / fit / plan`; read-only dependencies on matched data gates,
adaptive base identities/features/preprocessing, decoder_loss coordinates and
diagnostic counts, paired runtime, and the standard GELU constructor. Reviews
consume saved evidence. Production and historical trainers do not depend here.

`python -m pytest tests/test_resolved_fit.py -q -p no:cacheprovider`

Expected: reject non-training/duplicate/missing audit rows, mask unresolved
components, distinguish partial-row from whole-state success, preserve zero
identity, replay the saved GPU head exactly and keep objective-paired initial
weights identical. NumPy tests can run locally; GPU tests must run on the server.
