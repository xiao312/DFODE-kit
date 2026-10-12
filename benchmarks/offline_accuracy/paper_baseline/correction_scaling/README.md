# Direct correction: coverage and controlled scaling

## Responsibility and non-goals

Move from checked small-set fitting to diverse correction-training populations.
Keep `resolved_fit` and all historical results unchanged. This experiment uses
the continued GBCT base for each seed, not the original base used in that probe.
No CFD, new reference labels, test-set access, temperature switching, balancing,
capacity changes or shared-environment changes are part of the initial campaign.

## Frozen choices and gates

`plan.py` pins the two continued result hashes. Check their original-base lineage,
full artifact hashes and dataset identities. Freeze each base, and refit only
base-output feature scaling on its same 50k training prefix. Preserve its original
physical input scaling. Prepare immutable base predictions, 119 features and
FP64 correction targets for all 200k rows before selecting prefixes. Use fixed
1024-row padded inference batches to avoid batch-size-dependent FP32 kernel
changes between prefixes. Compare this arithmetic with the incumbent's saved
predictions and report any pass-decision differences. No target enters inference.

Decoder: `dhat=1e-14*sinh(asinh(base/1e-14)+c)` using the existing stable algebra.
Direct target: `c*=asinh(d/1e-14)-asinh(base/1e-14)`, no extra normalization.
The component rule remains `abs(dhat-d)<=1e-15+0.1*abs(d)`. A coordinate error
<=log1p(0.1) is sufficient in exact arithmetic, not necessary; score the physical
prediction and test that bound separately. Labels and reconstruction use FP64.

1. Coverage first: 10k/50k/200k training prefixes and fixed development. Check
   row uniqueness, finite values, disjoint snapshots, species/magnitude counts,
   positive/negative/zero counts, source diversity, temperature, pressure and
   fuel-composition ranges. Use separate temperature groups, not a sparse full
   cross-product. Report bins with development support but absent/sparse training
   support. Source-row IDs are split-local, not globally unique identifiers.
2. Compare ordinary and audited labels on the existing 16 train/16 development
   states in physical allowance units. Stop if any resolved audited component
   differs by >0.1 allowance. Retain all ordinary labels unchanged. Unaudited
   components are eligible nominal labels, not individually certified labels.
3. Regression gate: fit the 3x256 direct head to the same 16 checked training
   states around each continued base, using the historical 6k full-batch recipe.
   Require >=99% resolved and nontrivial acceptance in both seeds before scaling.
4. Initial size: fresh 3x256 zero-output GELU heads, direct-coordinate MSE versus
   physical log1p normalized error, both seeds. Same 10k rows, 6k updates, batch
   10k, fresh Adam, cosine 1e-4 to 1e-6, FP32 head, no clipping, final checkpoint.
   Record every 1k updates. Both objectives use all non-argon nominal components.
5. Scale to 50k then 200k only if direct improves development acceptance by at
   least 0.5 percentage points over its matched physical control AND over the
   frozen continued base in both seeds, with no worse nontrivial acceptance
   than either comparator. This is a progression criterion, not deployment
   acceptance. Inspect fixed group scores and tails before advancing. Retain
   paired physical controls at later rungs for interpretable size comparisons.
   If this fails, stop the automatic size progression and report the learning
   curves; do not silently widen, rebalance or extend the schedule.

Every rung has 60M presentations, not equal epochs or equal compute. The base
already saw 200k training states, and scalers use 50k; this measures correction
training requirements, not learning from scratch. Development has been reused
and is not an independent test. Optional balancing must have its own prospectively
declared protocol after a supported weak group is identified.

## Reporting and data-quality contract

Bins: exact zero, (0,1e-15], (1e-15,1e-14], (1e-14,1e-12], then upper bounds
1e-10,1e-8,1e-6,1e-4,infinity. Count species-state components and distinct states
separately for aggregated bins; within one species, one component equals one row.
Do not call a large component count broad independent source coverage. Mark
species-magnitude groups with fewer than 128 training states as sparse.

Report train/development acceptance, coordinate-error quantiles and sufficient-
bound counts, nontrivial acceptance, signs, tails, base pass-to-fail transitions,
species-magnitude/temperature groups, audited-subset scores and negative endpoints.
Group reference quality is unknown outside the existing audit, even when endpoint
spacing is small. Coverage cannot certify missing external operating regimes.
Save source/version/environment/cost evidence and exact model replay. No rounding
or positivity repair is added to hide failures.

## Interfaces and security

`python -m benchmarks.offline_accuracy.paper_baseline.correction_scaling.prepare
--campaign <200k> --previous <50k> --original <original-dataset>
--base-root <original-paired-root> --continued-first <seed11-continuation>
--continued-second <seed12-continuation> --output <new-prepared-directory>`
is read-only validation; add `--execute` to prepare GPU caches and coverage.

`python -m benchmarks.offline_accuracy.paper_baseline.correction_scaling.run
--prepared <prepared-directory> --seed 20261011 --size 10000 --objective direct
--probe-root <completed-probes-directory>
--output <new-fit-directory>` validates without mutation. `--probe` instead
runs the checked 16-state direct regression gate. Add `--execute` for training.
Run and review both probe seeds before any size rung. Select an idle GPU and set
CUBLAS_WORKSPACE_CONFIG=:4096:8 in the version-recorded project environment.
There is no CPU training fallback. Each fit is bounded to 1200 training seconds;
use an outer 1800-second command timeout including verification.

Only project-local ignored runs receive generated arrays, checkpoints and logs.
Existing files are never overwritten. No credentials or private mechanisms are
published. Checksummed prepared targets are immutable input, not checkpoint-local
recomputed labels. Fail closed on identity, finite-value or replay disagreement.

## Dependencies and verification

`prepare -> plan / coverage / model`; `run -> train / evaluate / model / plan`.
Read checked inputs via matched_work, continued weights via adaptive, and reuse
adaptive features, decoder_loss numerical coordinates and resolved_fit's probe
configuration and metrics. New code owns nominal-data scaling and group reports. Production and
historical modules never depend on it. The method catalogue is review-only.

`python -m pytest tests/test_correction_scaling.py -q -p no:cacheprovider`

Expect tests for exact bin boundaries, complete component partitions, source
diversity, label-quality gates, sufficient coordinate bound, paired initialization,
fixed-batch replay and GPU objective gradients. Reconcile saved predictions with
physical counts independently before publishing measured improvements.

Near-zero decoder agreement is measured in fixed physical allowance units
(difference <=1e-9 allowance), with identical physical pass decisions required.
This avoids a meaningless relative-only arithmetic check near zero. It does not
relax the physical acceptance rule. Historical resolved_fit code is unchanged.
