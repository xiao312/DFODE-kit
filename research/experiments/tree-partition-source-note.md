# GPU tree partitions: source check and bounded proposal

Research note, 2026-10-13. Documentation/source investigation only; no package
installation, training, environment change, or experiment registration was done.

## Decision and scientific scope

XGBoost single-output GPU histogram trees are an implementable control for
species-specific learned partitions. Use 58 independent boosters with a fixed
pseudo-Huber objective. The recommendation is to test one recipe on the two
existing base/data seeds, not to search architectures or objectives. Nothing in
the library evidence establishes improved chemistry accuracy.

The [species-distance manifest](EXP-20261013-species-distance.json) tests marginal
input relevance; the [local-function manifest](EXP-20261013-local-function.json)
uses 128 fixed centres and a smooth basis. [Recorded results](../RESULTS.md)
show better neighbor-transfer controls but negative trained local functions;
later independent neural species heads also fail transfer. Trees ask a different
question: can supervised, interacting input thresholds produce useful correction
regions? Historical RBF/MLP scores are context, not an equal-work causal comparison.
Pseudo-Huber also differs from their Smooth L1 objective. Do not attribute any
improvement exclusively to partition learning without a later matched control.

Do not create a second tree arm merely by multiplying each input by its positive
species-distance weight. In exact arithmetic that preserves each feature's order
and possible threshold partitions; only numerical binning effects may differ.
Independent target-specific split selection already supplies input relevance.

## Supported implementation and version boundary

- The documented GPU path is `tree_method="hist", device="cuda:<ordinal>"`.
  GPU-trained trees can be evaluated on CPU and conversely. QuantileDMatrix and
  GPU-resident `inplace_predict` are supported tools for this path.
  [Official GPU guide](https://xgboost.readthedocs.io/en/stable/gpu/index.html).
- In tag **v3.4.2**, both `reg:pseudohubererror` and `reg:absoluteerror` have
  device-aware gradient kernels. Pseudo-Huber uses
  `g=r/sqrt(1+(r/delta)^2)` and `h=(1+(r/delta)^2)^(-3/2)`.
  Thus its per-row gradient is bounded by fixed positive `delta` with unit
  weights; the Hessian shrinks in the tails. Absolute error now uses an adaptive
  smooth majorization scale and IRLS curvature, rather than the older median
  leaf-refresh algorithm.
  [Tagged objective source](https://github.com/dmlc/xgboost/blob/v3.4.2/src/objective/regression_obj.cu#L306),
  [current objective descriptions](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters).
- This is version-sensitive: **v3.0.5** absolute error uses sign gradients and
  median leaf updates, with explicit CUDA dispatch. It is also GPU capable;
  claims that absolute-error training is universally CPU-only are outdated.
  [Older objective source](https://github.com/dmlc/xgboost/blob/v3.0.5/src/objective/regression_obj.cu#L629),
  [CUDA leaf-update dispatch](https://github.com/dmlc/xgboost/blob/v3.0.5/src/objective/adaptive.h#L86).
- Multi-output remains documented as experimental with objective/metric
  limitations. Independent one-dimensional labels and 58 separately saved
  boosters avoid dependence on vector-leaf support or shared splits. Do not
  infer a universal GPU prohibition from old vector-leaf limitations; current
  multi-output development has moved on, but is unnecessary here.
  [Official multi-output guide](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html).

Proposed package pin: **XGBoost 3.4.2**, the stable patch release observed during
this review, with the exact wheel filename/hash, package variant, Python version,
build information and dependency lock recorded. This is conditional on a clean
GPU compatibility smoke test. Current documentation requires CUDA >=12.9; the
default Linux wheel uses CUDA 13, while `xgboost-cu12` is a CUDA-12 driver option.
Choose exactly one compatible pinned variant before registration. No silent CPU
fallback, nightly package or shared-environment upgrade. Older 3.0.5 is not an
automatic compatibility fallback: its release notes also mention CTK13 builds.
[Stable release](https://github.com/dmlc/xgboost/releases/tag/v3.4.2),
[installation variants](https://xgboost.readthedocs.io/en/stable/install.html#cuda-toolkit-variants-linux),
[3.0.5 release](https://github.com/dmlc/xgboost/releases/tag/v3.0.5).

## One prospective recipe

These are proposed engineering choices, not library-prescribed optimal settings.
Freeze them and the data identities in a manifest before looking at new scores.

| Item | Proposed fixed choice |
|---|---|
| Population | The already checked 50k training population per base seed; all 58 active species |
| Features | Existing 119 state/base features with frozen preprocessing and column order |
| Target | `z_j = (asinh(reference_j/1e-14) - asinh(base_j/1e-14)) / log(1.1)` |
| Learner | 58 independent `gbtree` boosters; sequential species order on one admitted GPU |
| Objective | `reg:pseudohubererror`, `huber_slope=1`, unit row weights |
| Initialization | Explicit `base_score=0.0`; no `base_margin`; zero correction reproduces base |
| Work | 256 boosting rounds, final checkpoint only; two existing seeds; 116 scalar fits total |
| Structure | `max_depth=4`, `grow_policy=depthwise`, `max_bin=256` |
| Regularization | `eta=0.05`, `min_child_weight=20`, `reg_lambda=1`, `reg_alpha=0`, `gamma=0` |
| Other | `max_delta_step=0`, `subsample=1`, all `colsample_* = 1`, `num_parallel_tree=1`, `nthread=1`, `validate_parameters=True` |
| Decode | Multiply tree output by `log(1.1)` and use the existing stable FP64 asinh correction decoder; preserve inactive AR policy |

The normalized target fixes the crossover without per-species label-derived
scales. Bounded gradients do not bound predictions or physical error; tiny tail
curvature can still make Newton steps problematic. `min_child_weight` bounds a
Hessian sum, not a minimum leaf row count. Record effective leaf sizes and tail
corrections. Explicit `base_score` prevents automatic intercept estimation.
[Parameter definitions](https://xgboost.readthedocs.io/en/stable/parameter.html).

Use frozen GBCT output coordinates only as a separately registered successor,
with the exact existing forward/inverse/normalization contract. Do not combine
coordinate choice, tree depth and objective into this first experiment. No
development-based early stopping, species selection, correction clipping,
temperature routing, reference lookup, new chemistry labels, or confirmation
access belongs to the proposed control. CatBoost adds no necessary capability
for this test, so no second-library comparison is recommended.

## Verification, replay and cost

1. Before real fitting, run a synthetic single-output GPU smoke test for the
   pinned objective, signed targets and zero initialization. Check recorded
   configuration actually names the selected CUDA device. Repeat a small fixed
   fit twice with identical data/order/configuration and compare model and
   prediction hashes. Fix seeds and row order; do not promise cross-device or
   cross-version bitwise training reproducibility. XGBoost's GPU histogram source
   uses fixed-point gradient accumulation, but that is not an end-to-end promise.
   [Histogram implementation](https://github.com/dmlc/xgboost/blob/v3.4.2/src/tree/gpu_hist/histogram.cu).
2. Save each booster using `save_model` as `.ubj` or `.json`, alongside
   `save_config`, feature/species order, base/scaler hashes, package identity and
   decoder configuration. Model IO is the supported portable archive; pickle
   memory snapshots are not a stable cross-version contract.
   [Model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
3. Fresh isolated inference receives states, frozen base/preprocessing and trees
   only. Compare saved versus reloaded GPU outputs, independently traverse a
   small sample of dumped trees, and compare explicit CPU replay numerically.
   Report physical pass-decision differences; CPU/GPU portability does not imply
   bitwise equality. GPU input to `inplace_predict` returns GPU output; include
   necessary transfers in end-to-end timing.
   [Prediction interface](https://xgboost.readthedocs.io/en/stable/prediction.html#in-place-prediction).
4. Apply the unchanged protected scorer to training, U20k and D1023, including
   repair/damage, sign, negative endpoints, severe events and all current guards.
   A lower coordinate loss is diagnostic only. A negative result closes this
   registered recipe; it does not justify a depth/bin/round scan.

Raw 50k-by-119 FP32 features occupy 23.8 MB (arithmetic estimate, not peak VRAM).
GPU memory additionally includes quantized ELLPACK, row work buffers and
histograms scaling with features, bins and nodes. Train species sequentially and
reuse/release data deliberately; do not create 58 concurrent training copies.
Use one QuantileDMatrix with explicitly replaced scalar labels if validated by
the smoke test. Peak VRAM and Python/CuPy allocator retention must be measured.
[GPU memory guide](https://xgboost.readthedocs.io/en/stable/gpu/index.html#memory-usage).

The declared maximum is 29,696 trees across both seeds, with at most 16 leaves
per depth-four tree; actual trees may be smaller. Record feature preparation,
quantization, all species fits, serialization and total allocation wall time
separately. Measure complete base + features + 58 predictions + decoder cost on
the contract batch, including synchronization and transfers, against the pinned
anchor and a same-implementation base. Many small predictor launches may exceed
the 1.5x cost guard even if training is quick. No runtime or VRAM prediction is
established by this source review.

Open prerequisites: actual host driver/wheel compatibility, pinned package
availability, repeatability on that host, full decoder/replay parity, and useful
repair-versus-damage behavior. These require the admitted smoke test and declared
experiment; they cannot be resolved from library documentation.
