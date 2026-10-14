# Reaction structure, affinity and attention

## Responsibility and non-goals

Test reaction-aware features and a small cross-attention correction under the
unchanged signed-increment contract. This is not an LLM, an ODE solver, a learned
mechanism, or a guarantee of thermodynamic consistency. Stoichiometry structures
message aggregation; it does not enforce conservation on returned increments.

## Interface, data and configuration

`gradient_probe.run --experiment EXP-20261014-reaction-attention` performs read-only
admission unless `--execute` is supplied. The JSON manifest fixes four factorial
arms: fixed pooling / learned attention, each without / with affinity. All use
separate reactant, product and collider maps from the pinned mechanism. Reaction
identity and reversible/type metadata are preserved. No net-stoichiometry-only
graph and no invented reaction-extent reference labels are used.

The worker consumes ordered T/P/Y, the frozen base, training references and the
mechanism. The first20k of the previous common50k are candidate-training states.
Scaler fitting uses only those20k. Replay receives states and model artifacts,
not labels. Outputs include metadata, training-only scalers, weights, histories,
predictions, live feature-computation timing, and independent physical scores.
The review records successful and archived interrupted worker time separately.
Worker wall time is not GPU allocation time. Recovery guards refuse ambiguous
artifacts or possible saved checkpoints; they do not authorize repeating fits.

Affinity is `-delta_gibbs/(R*T)` in matching J/kmol units. It is asinh-compressed
and standardized per reaction on training rows. Zero mole fractions use Cantera's
finite chemical-potential convention; these affinities are features, not certified
sign labels. Irreversible channels receive an explicit reversible flag, not an
invented reverse rate. No affinity-based sign gate is applied to finite increments.

The compact model forms reaction tokens from stoichiometric/collider-weighted
species covariates. Species queries aggregate them with fixed positive structural
weights, or two-head attention with those weights as a log prior. A residual MLP
and zero-initialized output return one correction coordinate per active species.
The checked fixed-asinh decoder remains unchanged. Attention has extra query/key
parameters; counts are reported and a win is not proof against capacity effects.
Every arm uses the same inputs except the declared affinity ablation. Full reaction
rates are NOT included. This separates thermodynamic information from rate features.

## Dependencies and security boundary

`gradient_probe.run -> prepare -> hidden_features.campaign.run_worker`
`worker -> correction_scaling.train -> model/chemistry`
`infer -> state-only model + mechanism -> saved increments`
`review -> pinned scorer + independent recount`

Candidate execution uses the existing Bubblewrap boundary. The mechanism is a
private model artifact, never committed. No credentials, network, confirmation
data, or reference arrays enter inference. Production has no dependency here.
GPU neural work uses the pinned environment; chemistry feature construction uses
one CPU thread. Feature computation is included in inference cost, not hidden by
a cache. Finiteness/axis failures stop execution; no clipping/normalization fallback.

Limit: reaction tokens average prepared participant covariates. This is a compact
first model, not a lossless representation of every reactant. Attention cannot
recover information removed by that averaging. Negative results apply to this
specific recipe. Neural fitting has a per-fit deadline and finite update count;
the controller monitors GPU contention, not a separate whole-worker deadline.

## Verification

`python -m pytest tests/test_reaction_attention.py -q`

`python -m pytest tests/test_reaction_attention_recovery.py tests/test_reaction_attention_retry.py -q`

Expected: stoichiometric orientation, collider handling, affinity units/identity,
training-only scaling, input validation, paired initialization, zero-base replay,
attention support, gradients, and save/reload tests pass. Synthetic checks do not
establish chemical accuracy. Run the registered campaign and then
`python -m benchmarks.offline_accuracy.paper_baseline.reaction_attention.review
--project <root> --run <run>` for full-population scoring and independent counts.

Source references: [Cantera3.2 kinetics](https://www.cantera.org/3.2/python/kinetics.html)
and [Torch2.8 attention](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html).
The implementation uses explicit deterministic attention operations for inspectable
weights and replay, with no dropout or mixed precision.

## Structural diagnostic

`python -m benchmarks.offline_accuracy.paper_baseline.reaction_attention.structure_review --project <root>`
reads the completed, hash-bound run under `EXP-20261014-reaction-information`.
It reports raw linear encoding ranks per species support, paired saved-score effects,
training curves and configuration parity. It fits no model and selects no checkpoint.
Full rank does not mean the trained downstream network can invert the encoding.
Check with `python -m pytest tests/test_reaction_structure_review.py -q`.
