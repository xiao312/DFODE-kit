# Correction decoder and loss: controlled diagnostic

Source and code check: 2026-10-12. This is a design note, not a training result.
It supplements the [adaptive protocol](../../benchmarks/offline_accuracy/paper_baseline/adaptive/README.md).
Keep that historical campaign unchanged. Do not claim that this experiment has
improved acceptance until saved predictions pass independent verification.

## Question and controls

Does failure come from weak pressure on large normalized errors, or from the
precision needed to cancel an incorrect base prediction? Test their interaction
with two decoders and two objectives, using two existing seeds: eight fits.

| Decoder | Mean log1p normalized absolute error | Mean normalized Huber error |
| --- | --- | --- |
| Base-relative linear, alpha=1 | A | B |
| Fixed-crossover asinh correction | C | D |

Use the same frozen verified GBCT/increment base, 200k rows, normalization
prefix, 119 input features, zero-initialized 3x256 correction network, paired
batches, 6k updates and final checkpoint. Keep the independent test closed.
Inspect A versus B first. Reuse an old A result only if its exact recipe and
verification match. Conditional alpha calibration is not part of this matrix.
An optimizer change made after observing failure is a separately named run.

The acceptance contract remains `tau(d)=1e-15+0.1*abs(d)`. It is not a decoder
scale, and it must not depend on the prediction. No species weighting, target
clipping, positivity projection or silent loss cap is added to one cell.

## Loss and gradients

Let `u=(prediction-reference)/tau(reference)`. The objectives are
`mean(log1p(abs(u)))` and `mean(Huber(u,0,delta=1))`. Huber is `u*u/2` for
`abs(u)<=1`, otherwise `abs(u)-1/2`. This agrees with the installed major/minor
version's [PyTorch 2.8 Huber documentation](https://docs.pytorch.org/docs/2.8/generated/torch.nn.HuberLoss.html).

For a scalar decoder D(c), away from zero error, the log-loss coordinate-gradient
magnitude is `abs(D'(c))/(tau+abs(error))`. For Huber it is
`min(abs(u),1)*abs(D'(c))/tau`. A mean contributes the same factor 1/N to both.
These are chain-rule calculations, not claims from the Huber documentation.
Bounded derivative with respect to u does not bound the parameter gradient.
Changing the decoder changes this gradient even when the loss is unchanged.

Record raw global parameter-gradient L2 norm, largest absolute gradient,
nonfinite counts and learning rate. Record clipping threshold and frequency if
clipping is enabled; zero clipping is also an explicit choice. PyTorch's
[gradient-norm API](https://docs.pytorch.org/docs/2.8/generated/torch.nn.utils.clip_grad_norm_.html)
returns the norm before clipping. Do not infer an entire run's norm distribution
from one batch at each 1k-update evaluation. Aggregate over updates or declare
the sampling schedule. Include the first update, when hidden-layer gradients
are initially zero because the final correction layer starts at zero.

## Decoders and stable arithmetic

For frozen b and crossover s=1e-14:

- Linear: `D(c)=b+(s+abs(b))*c`; exact target `c*=(d-b)/(s+abs(b))`.
- Asinh: `z_b=asinh(b/s)`, `D(c)=s*sinh(z_b+c)`;
  exact target `c*=asinh(d/s)-z_b`.

The linear derivative is `s+abs(b)`. At the correct answer the asinh derivative
is `hypot(s,d)`. The latter allocates local accuracy according to the corrected
magnitude, not an erroneous large base magnitude. It does not prove that the
correction function is easy to learn across the input domain.

The direct asinh decode can change b at c=0 through an encode/decode round trip.
Do not implement `where(c==0,b,direct_decode)`: selecting a constant at the exact
zero initialization removes the initial correction gradient.

A useful FP64 implementation uses this algebraic identity near c=0:

`D(c) = b + 2*b*sinh(c/2)**2 + hypot(b,s)*sinh(c)`.

It returns b at zero and has derivative `hypot(b,s)`. Use it, for example, for
`abs(c)<=0.5`. For larger corrections use the direct shifted-asinh formula. The
near-zero identity alone must not be used for large magnitude reductions: its
large terms can cancel. Both branches represent the same real-valued function;
test their boundary agreement, derivatives, signs, b=0 and small corrections.
This hybrid formula is our arithmetic adaptation, not a cited library algorithm.

Avoid evaluating unsafe inactive branches. Array `where` does not prevent
overflow in expressions evaluated before selection. Restrict inactive branch
arguments before evaluation or use masked indexing; do not clip active outputs.
Fail on nonfinite active predictions or gradients. FP64 decoder arithmetic must
be shared by fitting and checkpoint replay. Add exact zero identity and nonzero
initial-gradient tests on the actual GPU path.

## Oracle audit before training

For each decoder, compute c* from reference and frozen-base arrays in FP64,
cast through the actual FP32 correction output, and decode in FP64. Record
`abs(decoded-reference)/tau` by split, species and target-magnitude bin, including
maximum, quantiles and fraction above 1. Also retain an all-FP64 oracle result.
This is a label-based diagnostic, not an inference input or training result.
If coordinate normalization is later added, audit its full forward/inverse path.

Local allowed coordinate error is `tau/abs(D'(c*))`. The exact accepted interval
is more informative: linear endpoints are `(d +/- tau-b)/(s+abs(b))`; asinh
endpoints are `asinh((d +/- tau)/s)-z_b`. These intervals include zero crossings
and need not be symmetric around c*. Check neighboring FP32 numbers as well as
round-to-nearest before claiming that no representable correction can pass.
A failed single rounded oracle is not, on its own, such a proof.

For b=1e-8 and d=1e-12, tau=1.01e-13. A local NumPy FP64 calculation gives:

| Diagnostic | Linear | Fixed-crossover asinh |
| --- | ---: | ---: |
| Exact correction, approximately | -0.999899000101 | -9.210315372914 |
| Local allowed coordinate error | 1.009999e-5 | 0.10099495 |
| Error/tau after FP32 correction cast | 0.00293374 | 0.00000328167 |

Both rounded oracle corrections pass this example. Its implication is different
conditioning, not that FP32 linear correction is impossible. The exact asinh
accepted interval relative to c* is approximately [-0.1064663, 0.0962145].
Reproduce these checks on the full training/development arrays and CUDA kernels.

## Evidence and interpretation

Preserve the acceptance rule and report train/development scores separately.
Predeclare threshold pairs for `P(abs(prediction)>=eta | abs(reference)<=epsilon)`;
always report numerator and denominator, including empty bins. An empty bin is
not zero failure probability. An order-of-magnitude reduction is meaningful only
with enough baseline events. Use the same rows and compare paired transitions.

Also report normalized-error tails by species and magnitude, acceptance on
resolved nontrivial increments, sign errors above the absolute-error floor,
negative endpoints and base-pass-to-fail transitions. Under the current rule,
zero or opposite-sign predictions necessarily fail when `abs(d)>1e-15/0.9`.
Keep reference-resolution qualifications; a small reference is not automatically
either valid or negligible. No data expansion or CFD experiment is needed here.

A gain only on training suggests a generalization problem. No gain on training
motivates input, optimizer and residual-complexity checks, not a conclusion that
all representations fail. One failed Huber recipe does not reject Huber under
every optimizer. A decoder gain is a result for this frozen base and data domain.
Repeated development selection remains a limitation without independent tests.

Residual learning is motivated by [Wang and Lai](https://arxiv.org/abs/2307.08934),
who fit successive residuals in selected regression/PINN settings. The
[spectrum-informed follow-up](https://arxiv.org/html/2407.17213v1) emphasizes
residual variation and spectral initialization, not amplitude normalization
alone. Neither source establishes solver-level precision for this chemistry map.
The present two-by-two diagnostic is our targeted adaptation, not a reproduction
of either paper or evidence that logarithmic coordinates universally win.
