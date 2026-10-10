# Methods to improve chemistry-increment acceptance

For current canonical names and flowcharts, use the
[method catalogue](../research-method-catalogue.md). This note preserves the
literature rationale and the proposal at the time it was written.

Source check: 2026-10-09. This note separates published evidence from proposed
adaptations. It does not report a new training result.

## Decision and fixed target

The next objective is a substantial increase in acceptance, not another survey
of scalar transforms. Keep the current error contract fixed:

`E_i = abs(predicted_increment_i - reference_increment_i) / (1e-15 + 0.1 * abs(reference_increment_i))`.

A component passes when `E_i <= 1`. A complete state passes when all 58 scored
species pass. Report both. Do not improve the headline by loosening the tolerance,
removing difficult species, or rejecting samples without reporting coverage.

The completed development experiment reached about 36% component acceptance and
zero complete-state acceptance. The RMS-scaled residual recipe reduced the
original base's acceptance from about 32.4% to 23.6%. Small increments lost
accuracy. These are results for the tested recipes, not limits of residual
learning. See the [refinement protocol](../../benchmarks/offline_accuracy/refinement/README.md)
and [published results](https://github.com/xiao312/DFODE-kit/issues/3#issuecomment-6074397750).

## Ranked experiments

| Order | Adaptation | Why it fits this failure | Relative cost |
| --- | --- | --- | --- |
| 1 | Warm-start the best model; train directly on normalized physical error, with a state-level tail term | The pass rule depends on physical error and the worst species, not average coordinate loss | Low; no new labels or model |
| 2 | Learn corrections in an input-dependent local scale; start at exactly zero correction | The previous per-species RMS scale did not protect small increments within each species | Low to medium; one new correction model |
| 3 | Local interpolation/tabulation comparator in training-fitted coordinates | The current data occupy a restricted thermochemical region; a global MLP may be unnecessary | Low setup cost; potentially high query cost |
| 4 | Frequency-aware residual model with limited Fourier features | Published multistage methods use more than residual rescaling | Medium; requires a defensible feature scale |
| 5 | Learned choice between base and correction, with explicit risk/coverage accounting | Some corrections help while others damage an already good prediction | Medium; needs independent gate-training/calibration data |

These rankings are engineering judgments. They are not reported rankings from
the papers below.

## 1. Train for the errors that prevent acceptance

### Published evidence

Curi et al., *Adaptive Sampling for Stochastic Risk-Averse Learning* (NeurIPS
2020), optimize conditional value-at-risk (CVaR): loss in the worst fraction of
the population. Their algorithm also adapts sampling. This supplies a direct
precedent for training on difficult examples instead of only average loss. It
does not establish a chemistry accuracy guarantee. The
[full paper](https://proceedings.neurips.cc/paper/2020/file/0b6ace9e8971cf36f1782aa982a708db-Paper.pdf)
links the [author implementation](https://github.com/sebascuri/adacvar).

The inspected [CVaR implementation](https://github.com/sebascuri/adacvar/blob/master/adacvar/util/cvar.py)
uses a trainable threshold and positive excess above that threshold. Its
`SoftCVaR` uses `log(1 + exp(...))`; do not copy that expression unmodified for
our large normalized errors. Use a stable softplus if that formulation is used.

### Proposed adaptation

Warm-start the best transformed-state model. Keep its decoder. Replace or add
loss in physical units. Use both a component mean and a state-level tail term,
for example a smooth function of `max_i E_i`. A batch tail of component losses
alone can ignore which species jointly prevent a complete state from passing.

Begin with a moderate tail fraction and bounded weighting. Retain ordinary
sampling in every batch. A hard maximum from the first update can concentrate
all work on a few unresolved labels. A logarithmic penalty is stable, but its
gradient weakens for very large errors; measure this tradeoff rather than assume
that it matches the pass rule.

Use a small, fixed sweep of tail weight and learning rate. Select on an internal
training holdout. Keep the current development evaluation separate. Record
component and state acceptance, p90/p99/max normalized errors, and cost. This is
a CVaR-inspired objective, not a reproduction of the complete ADA-CVaR method.

## 2. Preserve small increments during correction

### Published evidence

Wang and Lai's MSNN method normalizes residual magnitude, but also changes the
first-layer frequency scale and activation. In its examples, residual rescaling
alone does not remove spectral bias. Its near-machine-precision demonstrations
use selected smooth regression/PINN problems, not a 58-output stiff chemistry
map. See [full text, Sections 2.2–2.3](https://arxiv.org/html/2307.08934v1).

The authors' inspected [FP64 regression example](https://github.com/YaoGroup/MultistageNN/blob/main/Regression/Reg1D_mNN_64bit_exp1.py)
uses ordered one-dimensional samples, estimates frequency from residual zero
crossings, combines Adam with L-BFGS, and adds samples at later stages. Thus, our
earlier FP32 GELU correction with 2,000 Adam updates was a simple residual
baseline, not a reproduction of their full recipe.

### Proposed adaptation

Freeze a competent base `b(x)`. Define a scale that is available at inference,
such as `s_i(x) = a/r + abs(b_i(x))`, and predict a correction in a local asinh
coordinate about the base. Initialize the correction output to zero. This
preserves the base at initialization and avoids assigning one absolute residual
scale to every magnitude of a species.

The base is not truth. Use the reference increment in the training/evaluation
error denominator; never replace it with the base prediction. A predicted scale
is a representation choice, not a new accuracy contract. If the base incorrectly
predicts a large increment, its local scale can still permit harmful corrections.

Add a training-only non-regression term on components that the frozen base
already gets right. Report a transition table: pass-to-pass, pass-to-fail,
fail-to-pass, and fail-to-fail. Zero initialization does not guarantee preservation
after training. Compare against equal-cost warm-start training without a second
model. Form and reconstruct corrections in FP64 and retain the arithmetic audit.

## 3. Use a local numerical comparator

### Published evidence

Pope's ISAT approximates repeatedly visited chemistry maps locally and grows a
table when reuse is not suitable. Its original paper explicitly states that the
estimated ellipsoids of accuracy provide good but imperfect local error control;
they are not a theorem that every retrieved point passes. See
[the original paper, Section 4.4](https://tcg.mae.cornell.edu/pubs/Pope_CTM_97.pdf).
Cornell provides an [official ISAT-CK7 source distribution](https://tcg.mae.cornell.edu/ISATCK7/).
This is a chemistry-specific precedent for local approximation and direct-solver
fallback, not a reason to promise the speedup from its original benchmark here.

For a smaller offline comparator, SciPy's
[RBFInterpolator documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
supports nearest-neighbor local interpolation and warns about conditioning,
polynomial rank, duplicate points, and the quadratic memory cost of a global
fit. Those constraints matter for nearly dependent species on a flame manifold.

### Proposed adaptation

Fit input scaling or a low-dimensional projection only on training data. Test
nearest-neighbor and local linear/RBF prediction with modest neighbor counts.
Use the same physical error contract and held-out rows as the neural models.
Record query cost, storage, and conditioning failures. Remove duplicates or
regularize explicitly; never hide a failed solve with an undocumented fallback.

Call this local interpolation, not ISAT. It lacks ISAT's flow-map sensitivities,
adaptive table logic, and online reference checks. A good result on correlated
snapshots from one flame would show useful local structure; it would not establish
transfer to another flame. Whole-trajectory/condition holdouts are still needed.

## 4. Add spectral information only where it has meaning

Ng, Wang, and Lai's spectrum-informed MSNN uses Fourier information to initialize
features and improves selected multidimensional regression examples. The full
text describes Fourier transforms on equally spaced one- and two-dimensional
grids. See [Sections 2–3](https://arxiv.org/html/2407.17213v1). I did not verify a
separate official implementation for this follow-up during this source check.

Our adaptation should first test a small Fourier-feature bank in standardized
inputs or local coordinates. Select feature bandwidth on training holdouts.
Do not FFT arbitrary dataset row order: it has no physical spatial meaning.
Do not claim that the paper's machine-precision result predicts performance on
irregular high-dimensional chemistry data. A spectral correction should follow,
not replace, the tolerance-scaled correction test above.

## 5. Gate corrections without hiding failures

SelectiveNet jointly learns prediction and rejection, including a tabular
regression example. It reports risk versus coverage, not universal accuracy at
full coverage. See [the full paper](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)
and [author code](https://github.com/geifmany/selectivenet). The inspected
[evaluation utility](https://github.com/geifmany/selectivenet/blob/master/selectivnet_utils.py)
computes loss on the selected subset. That selected-subset score must not be
confused with our unconditional acceptance score.

Proposed use: learn when a correction improves the base, using out-of-fold
training predictions and labels. Inference uses inputs and model outputs only.
Apply a conservative gate; otherwise retain the base. Report unconditional
acceptance and the correction coverage. An oracle gate that uses evaluation
labels is useful only as a diagnostic upper bound; it is not deployable. A later
solver-fallback version must report fallback frequency and full cost separately.

## Decision rules for the next bounded campaign

1. Freeze data identities, tolerances, seeds, and the base models before tuning.
2. Screen warm-start physical loss, local-scale correction, and local interpolation.
3. Use internal training holdouts to choose settings. Label the existing inspected
   snapshots as development evaluation, not an untouched test.
4. Prefer a clear improvement across seeds and magnitude bins, not one best run.
   Track the number of failing species per state even when state acceptance is zero.
5. Keep the 99% complete-state milestone unchanged. Record substantial intermediate
   gains, but do not call a component-rate increase a solver-level success.
6. If all recipes still fit training data poorly, improve optimization/features
   before assuming that more data alone will fix the problem. If training passes
   but held-out performance does not, examine coverage and increase data separately.

No model implementation, training, server mutation, or GitHub publication was
performed as part of this source note.

## Additional chemistry-specific input representation

Döppel and Votsmeier (2023) combine inverse-temperature/log-partial-pressure
inputs, a latent asinh representation and physical-space loss for steady surface
kinetics. Their examples use separate small networks for selected species, not
our full gas-phase finite-time flow map. The result supports testing input and
output conditioning together; it does not predict our achieved acceptance.
[Primary paper](https://pubs.rsc.org/en/content/articlehtml/2023/re/d3re00212h),
[primary supplement](https://www.rsc.org/suppdata/d3/re/d3re00212h/d3re00212h1.pdf).

Adaptation: use `1000/T`, `log(P/1 atm)` and a continuous log-like partial-pressure
coordinate with a fixed floor for zero species. Train independent small species
heads with a latent sinh decoder. Compare with local RBF on the same new inputs.
This changes several design factors; call it a bundled method candidate, then
separate the factors only if it works. Do not borrow the paper's speedups or
percentage errors as expectations for this chemistry dataset.

The paper's Sections 2.3.4–2.3.6 specify full-batch L-BFGS with strong-Wolfe
search and root-mean-square relative physical error. That is materially different
from short stochastic Adam with log1p loss. Test a bounded L-BFGS/RMS finish after
the same coordinate warmup; retain the Adam/log-loss candidate as a comparator.
Use mean per-species RMS budget error to balance independent output heads, and
record closure evaluations, memory and CPU cost. This is an optimization-plus-loss
comparison, not evidence that L-BFGS alone caused any difference.

## Mechanism-informed controls

Cantera exposes net species production and destruction rates in molar units.
Convert with molecular weight and initial density before predicting mass-fraction
increments. See the [official kinetics interface](https://www.cantera.org/stable/python/kinetics.html).
Our proposed controls are `h*f(Y)` and the analytic frozen scalar equation
`dy/dt = p-k*y`, reconstructed as `h*f(Y)*phi1(-h*k)`. Evaluate phi1 with
`-expm1(-h*k)/(h*k)` and its limit one at zero. The latter is only a diagonal
frozen-rate approximation: cross-species rate changes and conservation errors
remain. This is not the full Jacobian-based exponential Rosenbrock method.
It tests a possible physical starting point for learned residuals, not a neural
accuracy gain. Count actual mechanism evaluation in prediction cost.
