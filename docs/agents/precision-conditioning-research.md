# Precision-conditioned chemistry surrogates: research plan

Branch: `research/precision-conditioned-increments`

Starting revision: `b7d3943e805d9d074f25fa1b0de4c13b17bb711f`

Human review and the current decision log are centralized in
[GitHub issue #1](https://github.com/xiao312/DFODE-kit/issues/1).
This file is the reproducible technical plan, not an additional review inbox.

## Question and hypotheses

Can error-budget-aware coordinates, sequential residual models, and stable
reconstruction improve the accuracy-cost tradeoff of a chemistry surrogate over
direct signed-power increments and transformed-state increments?

H1: error-budget scaling plus asinh improves tail errors at a fixed model budget.
H2: additional residual stages improve physical errors beyond a larger single model
at comparable parameter count and inference cost.
H3: stable increment prediction and reconstruction preserve information otherwise
lost in endpoint subtraction or FP32 input quantization.
H4: a validated error estimator plus fallback can control rollout error economically.

These are hypotheses, not conclusions. Small magnitude and many correct digits are
separate metrics. Solver tolerances do not guarantee identical relative accuracy in
all increments or global trajectories.

## Existing code observations

- `training/train.py` builds FP32 features and transformed-state increment labels.
- `data/label.py` and the older `data/integration.py` path fix reactor tolerances at
  rtol=1e-6 and atol=1e-10; this is not a suitable precision reference by declaration.
- `training/positive_interval.py` retains physical arrays in FP64 and offers high/low
  thermochemical state inputs.
- `models/positive_interval.py` has `asinh-training-q90` for thermochemical delta-Y
  features. That feature representation is distinct from a species increment target
  representation and needs separate ablation.
- Existing signed-power enthalpy correction heads are distinct from sequential
  multistage residual training. Their names must not be conflated.

## Review checkpoints

The agent prepares code, bounded experiments, and reports. The researcher reviews
the evidence at each checkpoint and steers the next scientific decision.

1. Representation audit: inspect rounding, cancellation, and reconstruction error.
2. Reference/data audit: select mechanism, reactor constraints, state distribution,
   timestep range, tolerance ladder, and train/validation/test trajectory splits.
3. Small paired ablation: review learning curves, failures, physical errors and cost.
4. Residual stages: review improvement per stage and compare equal-budget controls.
5. Rollouts: review conservation, positivity, ignition/trajectory accuracy and fallback.

Each checkpoint has a short decision note, reproducible configuration, machine-readable
metrics, figures and one canonical HTML report. Unexpected findings are reported
before escalating experiment scale. No large training sweep precedes checkpoint 2.

## First implementation

`benchmarks/precision_conditioning/` provides an independent NumPy-only audit.
Known synthetic increments avoid pretending that subtraction from rounded endpoints
is an exact reference. It reports relative and error-budget-normalized error, signed
zero loss, and magnitude bins. This establishes a numerical baseline only.

## Chemistry reference design

Start provisionally with a small H2 mechanism to make reference checks inexpensive,
then test the selected ammonia/methane mechanism. Confirm this choice at checkpoint 2.
Record constant-pressure versus constant-volume, energy equation, mechanism checksum,
species order, solver/library versions and actual tolerances. Tighten tolerances over
a ladder and compare outputs; call differences a reference uncertainty estimate rather
than an exact bound. A second solver or higher precision reference is desirable for
claims near the FP64 floor. Track the difference between integrating a small change
and subtracting two stored endpoints. Some increments cannot be identified reliably
from endpoint labels alone.

Split by parent trajectory/case before deriving correlated interval pairs. Fit
normalization and scales on training only. Preserve untouched test trajectories and
test both interpolation and explicit out-of-distribution conditions.

## Ablation matrix

Keep dataset, split, seeds, optimizer, architecture budget and compute budget fixed:

| Factor | Initial variants |
| --- | --- |
| Target | transformed-state delta; signed-power physical delta; budget-linear; budget-asinh |
| Precision | full FP64 reference baseline; explicit mixed precision; FP32 baseline |
| Approximation | one stage; two frozen-base residual stages; equal-budget larger single model |
| Reconstruction | physical increment plus state; measured high/low or compensated alternatives |
| Loss | transformed-space baseline; physical error divided by state error budget |

Report per-species and per-magnitude errors, relative errors only with a declared
floor/mask, weighted RMS and maximum errors, budget exceedance, positivity,
element/mass conservation, temperature/enthalpy consistency, wall time and inference
cost. A tiny absolute error alone is not evidence of significant digits.

Residual stages fit the remaining physical increment after freezing earlier stages.
Train-only residual statistics determine their scales. Accumulate in FP64 and retain
the increment separately from the rounded state. A compensated accumulator is useful
only if the downstream interface preserves its compensation across steps.

## Integration validation

One-step held-out accuracy comes before adaptive rollout. Compare against the reference
solver on fixed intervals and full trajectories. Test timestep sensitivity, semigroup
consistency, ignition delay, extinction if relevant, positivity, elemental conservation
and thermal consistency. Calibrate an estimator on held-out data; use step rejection,
smaller intervals or solver fallback when it exceeds the budget. Report the estimator's
missed-error rate and fallback cost. Conservation projection and normalization must
have their own ablations because they can mask or introduce increment errors.

## Asset organization

One project checkout contains source and portable experiment configs. Generated assets
live under ignored `runs/precision-conditioning/<run-id>/` with `config.json`, source
revision, environment metadata, metrics, figures, checkpoints and `report.html`.
Shared large inputs have checksum manifests and references rather than duplicate copies.
Machine paths and credentials stay outside published configs. Keep workstation and
server source revisions matched through Git; transfer only selected review artifacts.

## Sources and claims to verify

- [Direct increment transformation study](https://arxiv.org/html/2507.08277v2).
- The user-provided GBCT discussion needs its exact repository/paper before reproduction;
  it must not be silently treated as the DFODE-kit baseline.
- This plan implements the supplied research direction; it does not claim an exhaustive
  literature review or novelty assessment.
