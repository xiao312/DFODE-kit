# Reaction structure, affinity and attention: reviewed results

Implemented and tested reaction-aware inputs and a small transformer-style
cross-attention head. All16 GPU fits and full-population checks are complete.
No candidate was promoted. Continued GBCT remains the incumbent.

## What changed

1. Separate reactant, product and collider maps describe reaction structure.
2. Optional affinity gives the input state's thermodynamic driving force.
3. Fixed pooling or two-head attention combines reaction features for each species.
4. A learned correction uses the same fixed-asinh decoder and physical pass rule.

This structure does not enforce conservation on the returned increment. The model
does not read reference labels at inference or switch by temperature.

## What the experiments show

- In the first comparison, attention beats pooling in12/12 matched aggregate
  comparisons by0.078-0.275 percentage points. It has2,112 more parameters, so this
  is not a pure attention-versus-capacity result.
- Affinity improves9/12 comparisons, with effects from-0.029 to+0.148 points.
  There is no strong, consistent affinity gain.
- Four times the training work improves fitting slightly. Same-source acceptance
  reaches only72.38-72.51%, versus about72.24-72.27% for the base. Damage, tails,
  negative endpoints and cost still fail the protected rules.
- A fixed10% uniform global prior does not improve development acceptance.
  The repeated local control matches historical predictions exactly. The10% is
  prior mass, not a measured fraction of attention on distant reactions.

Some development scores improve by about one percentage point. That is not enough:
the correction damages about4-5% of base-passing components and costs several
times as much to run. The permitted damage is2%; the latency limit is1.5x.

## Decision

Reject these candidates. Close the tested work extension and fixed global-prior
choice. Do not infer that all reaction-aware models or transformers are ineffective.
Next, examine reaction-resolved kinetic scales on training data before registering
another model. Affinity and kinetic rate scale are different inputs.

The owner-authorized GPU UUID repair preserves evaluator isolation and scoring.
Other users' workloads were not changed. Interrupted attempts remain archived.
47 focused server tests and4 catalogue tests pass. No confirmation data was opened.

See [current results](https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/research/RESULTS.md)
for both seeds and populations, and [method diagrams](https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/docs/research-method-catalogue.md#pool-state)
for the inference sequence. These GitHub views do not require a new Pages build.
