# Gradient interference: source check and a bounded diagnostic

Status: methodological note, not an executed experiment or a candidate claim.
Scope: the current signed-increment contract; no new labels or inference rules.

## What the primary sources establish

PCGrad uses parameter gradients of task losses. When their dot product is
negative, it removes the conflicting projection; it then combines the modified
gradients. Algorithm 1 uses random task order for multiple tasks. The authors
allow the result to enter Adam. Their explanation involves conflict, unequal
gradient magnitudes, and curvature: negative cosine alone is not sufficient
evidence of harmful optimization. The two-task convex theorem also permits
stalling at exactly opposing gradients. The one-step loss comparison has
curvature and step-size assumptions. Appendix A.4 analyzes a strongly convex,
smooth heavy-ball setting, not general Adam. None of these results proves
chemical-increment acceptance or preservation on unseen states.
[PCGrad, Sections 2–3 and Appendix A](https://arxiv.org/html/2001.06782v3)

Adam combines current gradients with stored first and second moments and applies
coordinatewise scaling with bias correction. Thus its actual displacement is
not generally a scalar multiple of the current gradient. The optimizer state,
parameter ordering, learning rate, epsilon, decay and implementation settings
must be preserved in a counterfactual step.
[PyTorch 2.8 Adam algorithm and state documentation](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)

## Adaptation to this project: deductions, not published findings

Let P and F be fixed training-component masks for incumbent passes and failures
under the unchanged physical contract. These are two parts of one regression
problem, not evidence of two natural tasks. They require no task identifier at
inference. Do not recompute membership from the candidate's current predictions.

For a componentwise loss, output gradients for disjoint P/F masks occupy disjoint
entries of the same output tensor. Their dot product is zero by construction.
This does not rule out interference after backpropagation: with output Jacobian
J, parameter gradients are J-transpose times those output gradients. Compute
their dot product in the same shared parameter vector, not across unrelated
output entries or across species merely because they share a name.

If species have disjoint coefficients, different species have no shared-parameter
conflict in that layer. Pass/fail components within a species can still conflict.
Report whole-model and shared-block diagnostics separately when sharing exists.

Preserving a group's differentiable loss is not preserving each component's
acceptance. A preservation-interval penalty can also have zero gradient while
all passes remain inside their intervals; zero is not evidence of no future risk.

## Proposed training-only diagnostic before a new optimizer family

Register this diagnostic and its finite work before execution. Use both primary
seeds, the existing initial and final candidate checkpoints, and 16 fixed batches
of 2,048 training rows selected by ID before examining their errors. Use a fixed,
disjoint training-only check batch for step transfer. If a checkpoint lacks Adam
state, mark the resumed-Adam comparison unavailable; a reset is a different test.

1. Fix P/F from the frozen incumbent and training references. Retain the current
   scalar loss and reduction. For a population mean, define each group term as
   its masked loss sum divided by the total eligible batch count. Verify that
   g_P + g_F reproduces the original data-loss gradient. Keep regularization
   separate and apply it identically; equal group means would change weighting.
2. Record group counts, gradient norms, dot products and cosine per shared block
   and in aggregate. Use a declared numerical zero threshold; empty or zero-norm
   groups have undefined cosine, not zero conflict. Repeat across batches rather
   than selecting one extreme pair. Accumulate diagnostic dot products in FP64.
3. Clone model and full Adam state twice at each checkpoint/batch. Compare one
   ordinary step with one two-group PCGrad step, using the same learning rate,
   regularization and data. Never chain these trial steps or mutate saved fits.
   Preserve parameter ordering and assert that restored control replay matches.
4. Save actual displacements Delta-theta. For each group, compare
   g_group dot Delta-theta with its measured post-step loss change. Record step
   norms, finite values and the nonlinear remainder; do not infer Adam effects
   from raw cosine. A smaller step can confound a claimed projection benefit.
5. On both the step batch and fixed training check batch, decode physical
   predictions and count repairs, damage, signs and acceptance using the original
   masks and tolerance. Retain original base-error margins for damaged passes.
   Do not use development labels to choose batches, weights or step settings.

Proceed to one controlled training comparison only if harmful parameter-step
interactions recur and projection shows a repeatable repair/preservation benefit
on the training check batch, not just lower group loss. If the effect depends
only on a smaller displacement, require a step-size-matched control before crediting
gradient direction. If conflict is absent, or projected steps still damage passes,
do not start a PCGrad sweep. Diagnose parameter sharing, objective geometry or
local prediction structure instead. Even a favorable diagnostic only motivates
a trial: full protected U/D scoring, replay, latency and all promotion gates remain
mandatory. No chemistry effectiveness or unseen preservation is established here.
