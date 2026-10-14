# Adaptive residual representation: bounded follow-up

Source check: 2026-10-10. This is a proposed experiment, not a result or a
guarantee of higher acceptance. Use the
[method catalogue](../research-method-catalogue.md) for canonical method names.

## Decision

Use the verified GBCT target with increment-scaled loss as the frozen starting
point. Keep the same 200,000 training rows, development rows, two seeds, species,
and tolerance grids. Keep the independent test closed. Add one correction stage,
not a new dataset or CFD campaign.

The question is: **Does a representation fitted to the remaining error improve
acceptance more than extra training or a correction with a fixed scale?**

An improvement is an experimental objective. It is not implied by the word
adaptive. Preserve all 24 matched-target results as historical controls.

## What the sources support

Wang and Lai train successive networks on normalized residuals. Their regression
method uses residual RMS for amplitude and changes first-layer settings to
represent higher frequencies. Normalizing residual magnitude alone is not their
complete method. Their examples motivate a staged approximation; they do not
establish accuracy for our high-dimensional chemistry map.
[Primary paper, Sections 2.2–2.3](https://arxiv.org/html/2307.08934v1).

Ng, Wang, and Lai use spectral information to initialize later stages. They
identify limits of the earlier approach in multiple dimensions. Their Fourier
construction uses ordered, equally spaced spatial grids. Our chemistry samples
are irregular points in thermochemical space. Do not apply an FFT to dataset row
order or call a generic Fourier-feature network a reproduction of this method.
This paper supports checking residual complexity, not assuming that rescaling
solves it.
[Primary paper, Sections 1.2, 2.1 and 3.4](https://arxiv.org/html/2407.17213v1).

Both studies assume exact or noise-free regression data in their precision
examples. Chemistry labels have numerical uncertainty. Later stages must not be
interpreted as recovery of digits that the reference calculation did not resolve.

## What is already implemented

| Existing path | Existing function | Difference in this proposal |
| --- | --- | --- |
| [refinement](../../benchmarks/offline_accuracy/refinement/README.md) | Frozen base plus a physical residual scaled by one training RMS per species | Combine residual calibration with an input-dependent local scale |
| [improve/coordinates.py](../../benchmarks/offline_accuracy/improve/coordinates.py) | Base-relative asinh correction; zero correction preserves the base | Fit a correction scale from the new base's training residuals, rather than use only its predicted magnitude |
| [improve/neural.py](../../benchmarks/offline_accuracy/improve/neural.py) | Small earlier correction and continuation experiments | Use the stronger verified GPU baseline, fixed 200k data, paired controls and measured cost |
| [matched_targets](../../benchmarks/offline_accuracy/paper_baseline/matched_targets/README.md) | Four global targets and three objectives under matched conditions | Predict the remaining physical error around a frozen result, rather than relearn the entire increment |

Call this a **residual-calibrated local correction**. Do not describe it as our
first residual model, the first adaptive representation, or an exact MSNN
reproduction.

## Proposed representation

Let `b_i(x)` be the frozen base prediction and `d_i` the training reference.
Compute the physical residual `e_i = d_i - b_i` in FP64. Use a positive local
scale available at inference, for example:

`s_i(x) = a/r + abs(b_i(x))`.

Fit a positive factor per species and predicted-magnitude bin from training
residuals only. The implementation selects the conditional 75th percentile,
rather than one global RMS factor:

`alpha_i(bin) = quantile_0.75_train(abs(e_i) / s_i(x), bin)`.

Use the same first 50k training rows for this calibration and feature scaling.
Use eight fixed bins with edges 0,2,...,16 in log10(1+abs(base)/1e-14).
Fewer than 128 samples falls back to the species' global 75th percentile.
Bound factors to [.01,100], including zero-residual species. Interpolate log
factors between bin centers, with constant endpoint extrapolation. Record raw
values and bin counts. This robust conditional scale is our adaptation, not a
literal MSNN recipe. The [module protocol](../../benchmarks/offline_accuracy/paper_baseline/adaptive/README.md)
and executable `plan.py` are the source of exact run choices.

Use `s_i(x) * alpha_i` as the physical scale for a zero-initialized correction.
A linear residual decoder is the simplest isolated first test:

`prediction_i = b_i(x) + s_i(x) * alpha_i * correction_i(x)`.

A local asinh decoder is an alternative, but changing both the decoder and scale
would mix two factors. Select one decoder and use it in both correction arms.
Preserve exact zero-correction identity. Fit and freeze any extra feature
normalization on training rows. The network can receive ordinary inputs plus a
signed, compressed base prediction; both correction arms must receive the same
features. It must never receive the true residual at inference.

This scale is a representation, **not an error allowance**. The primary score
still uses `abs(error)/(1e-15 + 0.1*abs(reference increment))`. Keep state-scaled
evaluation separate. A train-fitted scale is not a prediction of uncertainty.

## First bounded experiment

Use three arms for each existing seed: six fits in total.

1. **Continuation control:** continue the GBCT-increment model with the same
   physical loss for the extra budget.
2. **Fixed local correction:** freeze the base; use the local correction above
   with `alpha_i = 1`.
3. **Residual-calibrated local correction:** freeze the same base; use the
   training-fitted `alpha_i`.

Start with 6,000 additional updates, batch size 10,000, paired row order and a
predeclared learning-rate schedule. Reset Adam explicitly for all three arms;
call the first arm a warm-start control, not uninterrupted optimizer continuation.
Use equal correction architecture, initialization and features in arms 2 and 3.
Use the same final physical loss without adding a tail penalty in only one arm.
Save final checkpoints; do not select different stopping points from development
scores. If a later run changes any of these choices, record the change before
execution.

Equal updates are not equal cost. Report extra training seconds, total base plus
correction cost, parameter counts, peak memory, and end-to-end batched inference
time. Extra base-output features and a second network are part of the method's
cost. This is not a perfectly parameter-matched architecture comparison.

## Evidence and stop rules

Before training, test `base + (reference - base)` reconstruction in FP64, zero
correction identity, finite gradients, frozen base hashes and saved-model replay.
Reject nonfinite values rather than hide them by clipping. Preserve the existing
reference-uncertainty limitations.

Report train and development results for each seed: component acceptance,
whole-state acceptance, normalized-error p99/max, failing species per state,
negative endpoints, conservation checks, and pass-to-fail/fail-to-pass counts.
Show both error-scale policies and the unchanged zero-update control.

A useful predeclared development milestone is at least five percentage points
more component acceptance than the frozen base in both seeds, with lower p99
error and no increase in negative endpoints. This is an engineering target, not
a statistically established threshold. To attribute the gain to residual scale
calibration, it must also beat the fixed-scale correction. To justify added
complexity, compare its gain and cost with warm-start training. Do not call a
component-rate gain solver-level reliability when complete states still fail.

If training improves but development does not, stop adding stages and inspect
generalization. If both remain poor, inspect residual features and optimization.
Only then consider revised local scale bins or frequency features with training-only
selection. Do not expand the dataset, loosen tolerances, or open the test to make
this experiment appear successful.
