# Selective correction: optional source-backed diagnostic

Status: not selected, trained or qualified. The fixed base remains the fallback.

## Existing implementation check

A search of `benchmarks/offline_accuracy`, broader benchmark Python sources and
the current evidence ledger found no learned base/candidate inference gate.
Existing `research_contract.gates` and `correction_scaling.gate` check experiment
qualification or progression. `hidden_features/preservation.py` is a training
penalty, not a selector. `conditional_calibration` fits output-bin corrections,
not a decision between two frozen predictions. This is a scoped search finding,
not a claim about private branches or artifacts that were not inspected.

## Primary-source findings

Mozannar and Sontag formulate the loss of a predictor plus deferral rule using
the costs of both possible decisions. They reduce expert deferral to cost-sensitive
learning and give a consistent surrogate; consistency concerns the population
minimizer over an unrestricted function class. Their classification results do
not guarantee that a finite fitted gate preserves chemical increments. The useful
adaptation is to learn comparative outcomes, not confidence in one model alone.
[ICML 2020, Sections 3–5](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf)

Geifman and El-Yaniv distinguish selective risk from coverage and derive a
threshold-selection risk bound under an i.i.d. sampling assumption. Their rule
can return zero coverage when the target cannot be met. This is not a guarantee
under arbitrary distribution shift or a guarantee that an imperfect fallback
will preserve total-system accuracy. The chemistry data contain dependent
components and few source snapshots; treating every component as an independent
Bernoulli observation would not establish the paper's assumptions.
[NeurIPS 2017, Sections 2–3](https://proceedings.neurips.cc/paper_files/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)

## Project deductions and labels

Freeze base b and corrected prediction c. Use the unchanged tolerance to compute
their training pass indicators B and C. The selector g chooses c when g=1,
otherwise b. Use an exact branch operation, not arithmetic that can introduce
rounding or propagate an unselected invalid value. References define training
labels only: repair=(not B and C), damage=(B and not C), same-status=otherwise.

The selected net change is sum(g*repair)-sum(g*damage). Its damage rate is
sum(g*damage)/sum(B), not damage divided by selected rows. A learned estimate
of repair probability minus damage probability targets net passes; it does not
enforce the separate preservation limit. Both-pass and both-fail selections can
still change error tails, signs or endpoints. Do not discard those components
from final scoring merely because their pass status is unchanged.

This is selective replacement, not abstention: every state still receives an
increment. It is not a numerical certificate. Species-wise selection can also
break conservation even when each original vector conserves mass; report mass
and elemental residuals as well as the protected metrics. No projection is implied.

## Smallest proposed training-only discrimination

Question: do permitted base/candidate features predict transferable repair versus
damage better than always keeping the base or always using the correction?

Prefer one previously tested 10k correction, selected before this diagnostic.
Freeze both models. Select 12k existing training-pool rows outside that correction's
fitting rows and outside protected U/D. Verify all preprocessing exposure; the
base may have seen them. If such rows are unavailable, obtain correction-out-of-fold
predictions under a separately registered fit; merely cross-fitting the gate does
not remove optimistic labels from an overfitted correction.

Use three fixed 4k state folds, stratified by source, rotating fit/calibration/check
roles. Keep all species of a state in the same fold. Fit one species-specific
three-class affine softmax model on four permitted features: asinh(b/s), asinh(c/s),
their absolute difference and their sign agreement, with fixed s=1e-14. Use one
declared regularizer and finite fitting budget; no class rebalancing or loss sweep.
These modest features test output disagreement without a temperature rule.

On the calibration fold only, set a threshold on estimated repair-minus-damage
probability to maximize net passes subject to <=2% empirical damage of base passes.
Predeclare ties in favor of fewer replacements and include always-base as fallback.
Freeze the threshold before the check fold. This is empirical calibration, not a
statistical risk guarantee. Do not tune features or thresholds from U/D results.

On untouched check folds, report selected count, repair/damage with denominators,
overall/nontrivial acceptance, species/source cuts, signs, tails and validity.
Compare always-base, always-candidate and learned selection on exactly those rows.
Treat cross-fit results as diagnostics: fold-specific gates are not one deployable
candidate. Proceed to a preregistered final gate only if repair survives without
comparable damage; otherwise close this limited feature hypothesis, not all gating.

Any final candidate uses the same declared function everywhere and only state,
base/candidate outputs and frozen gate artifacts at inference. Count the complete
base, correction, features and gate in latency; gating after correction does not
save its compute. Full protected U/D gates and label-free replay remain mandatory.
No source establishes chemical effectiveness or shifted-domain safety here.
