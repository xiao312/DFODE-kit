# Evaluation contract v0.1 (proposed)

## 1. Binding and provenance

Every evaluated result must carry: experiment ID; hypothesis ID; source commit; scorer version/hash; frozen anchor and current incumbent IDs; mechanism/hash; chemistry interval and reactor assumptions; reference version; split ID/hash; ordered species list; ordered row IDs; precision/runtime settings; parameter counts; model weights hash; inference signature; training, preparation and evaluation cost; and all prespecified seeds, including failed runs.

Predictions must be reconstructed from state inputs and saved model artifacts. Reference arrays belong to the evaluator, not the inference interface. Checking a hash in a worker-writable file is not an access-control boundary. A trusted process or separate account must pin the scorer and dataset manifests and own sealed evaluation assets.

Never drop failed or nonfinite predictions from denominators. Malformed or nonfinite references invalidate the dataset binding and require an explicit reference repair, not silent filtering. Resolution masks depend only on the fixed reference evidence, never on candidate error. If a population is not independently qualified, report nominal scores and separately qualified-subset scores, including coverage.

## 2. Populations and permitted uses

**T: training.** Use the current candidate-training subset for optimization. Report the base's prior exposure, feature-calibration exposure, and candidate exposure separately.

**U: correction-unseen same-source diagnostics.** Preserve the 20k fixed row IDs already selected from four base-training snapshots. These rows were not used to fit the current 10k correction but the base saw them, and some occurred in calibration. This is no longer a fresh research holdout after repeated analysis. If later admitted to candidate training, version its role and do not continue calling it correction-unseen.

**D: development.** Preserve the 1,023 existing rows. They are repeatedly inspected research data, not an independent test. Do not rename them after an apparently good result.

**R: regression and challenge suite.** Include the checked small-set feasibility cases, targeted sign-reversal audit, weak-update cases, and analytically generated scorer/decoder edge cases. Label previously inspected challenges as known regression cases. The 16-state training probe is not evidence of generalization. Synthetic scalar arithmetic tests are not chemistry-accuracy experiments.

**C: confirmation.** A separate custodian selects an independent flame realization or prespecified operating-condition case inside the declared target domain. Candidate weights and all preprocessing are frozen before access. The baseline is evaluated on the same case by the custodian. A single finalist per research epoch may be submitted; no new finalist may be tuned against those same results while preserving a fresh-test claim. Log every submission. Once used for selection, a confirmation set becomes development data for future work. Inaccessible raw labels and limited feedback reduce leakage; neither ensures indefinite statistical independence.

All model fitting, threshold choice, feature-layer choice and recipe search use T and explicitly declared development information. Population shifts, new cases and changed row memberships require new manifests.

## 3. Primary numerical quantities

Let a=1e-15, r=0.1 and tau_i=a+r*abs(d_i). All indices below refer to active species-state components, with fixed rows and species order.

    E_i = abs(pred_i - d_i)/tau_i
    A = mean(E_i <= 1)

Calculate component acceptance separately for each population, seed and slice. Calculate all-active-species state acceptance separately by row. Do not pool all cells into a single WRMS test; WRMS is not this research contract.

The nontrivial set is N={i:abs(d_i)>a/(1-r)}. Count zero predictions as wrong sign on N. Report both nominal nontrivial results and a resolution-qualified subset. A qualified sign claim requires stable reference sign and a reference uncertainty small enough to resolve the sign; where only an empirical bound is available, retain that qualification.

Report p50, p95, p99 and max of E, plus rates E>10, E>100 and E>1000. Quantiles cannot replace exceedance counts. For same-sign nontrivial predictions, also report abs(log10(abs(pred)/abs(d))); count omitted sign/zero failures separately so this diagnostic cannot hide them.

## 4. Small-target and activity preservation

Retain the historically declared event grid:

    epsilon in [1e-15, 1e-12, 1e-9]
    eta in [1e-12, 1e-8, 1e-6], eta > epsilon
    events = count(abs(d)<=epsilon AND abs(pred)>=eta)
    denominator = count(abs(d)<=epsilon)

The primary historical pair is epsilon=1e-12, eta=1e-8. With zero baseline events, a tenfold reduction is NOT ASSESSABLE. Do not report it as achieved or replace the threshold post hoc. Report raw counts and denominators. These measured rates on correlated components are not independent Bernoulli population guarantees.

An always-zero model is a mandatory negative control. It may pass floor-dominated cases and must fail the active population. Never declare success on small-target specificity alone.

## 5. Repair, preservation and anchor protection

For a fixed comparator, define P0=(E_base<=1) and P=(E_candidate<=1):

    repaired = count(not P0 AND P)
    damaged = count(P0 AND not P)
    R = repaired/count(not P0)
    Dmg = damaged/count(P0)
    A_new = A_base + (1-A_base)*R - A_base*Dmg

Return null, not zero, for an empty denominator. Test the identity independently. Split damaged components by initial base E in [0,.1], (.1,.5], (.5,.8], (.8,1]. For asinh correction candidates, optional acceptance-interval diagnostics distinguish wrong-direction changes and overshoot; do not force that decoder on other architectures.

Maintain both a moving incumbent and a fixed campaign anchor. Promotion guardrails compare against BOTH, so a series of locally tolerated regressions cannot erase a species or regime's original accuracy. A scientific model may repair some components and damage others; a candidate is not improved merely because its repair percentage exceeds its damage percentage, since their denominators differ.

## 6. Slice registry

Use three primary magnitude regimes: floor-dominated abs(d)<=a/(1-r); weak nontrivial a/(1-r)<abs(d)<=1e-12; active abs(d)>1e-12. These boundaries are prospective protocol choices, not fitted to candidate results. Keep existing finer historical bins unchanged alongside them.

Also report every active species, each source snapshot, true increment sign, temperature groups, and existing species-by-magnitude groups. Do not average away a weak group. Micro acceptance weights components equally; a macro group score uses a FROZEN list of sufficiently supported groups and fixed weights. Do not derive the macro weights from the candidate's successes or silently remove sparse groups.

For the proposed pilot, a group with at least 128 eligible components is large enough to apply the coarse automatic group gate. This is a practical reporting threshold, NOT a statistical sufficiency claim. Smaller groups remain in micro scores and in sparse-group tables, with exact counts. Missing required primary-regime evidence makes confirmation NOT ASSESSABLE; a planner may not declare those regimes solved.

Temperature groups are reporting slices, not inference switches. Source diversity and reference quality are separate from component count. Include coverage summaries for state, source, sign, magnitude and species.

## 7. Proposed candidate-promotion policy

These values are recommendations to be approved ONCE before the campaign. They are not literature-derived safety standards and do not retroactively alter historical gates.

For every prespecified seed on both U and D, require:

- Absolute micro acceptance gain >=0.005 versus the current incumbent AND fixed anchor.
- Nontrivial acceptance no lower than either comparator.
- Primary weak and active group acceptance no more than 0.005 below either comparator; supported per-species acceptance no more than 0.01 below either comparator. Broader improvements remain the objective; these are maximum pilot tolerances, not an invitation to sacrifice groups.
- Damage rate on comparator-passing components <=0.02 versus BOTH comparators.
- Nontrivial sign-error count no higher than either comparator on fixed masks.
- Global and weak-group normalized p99 no more than 1.05 times either comparator; preserve zeros exactly in ratio comparisons and report NOT ASSESSABLE for absent groups rather than invent a denominator.
- No new observed historical severe-event failures where comparator count is zero. Where counts are positive, no increase. Improvement claims need adequate event support, not merely two seeds on the same states.
- Finite returned increments for every row, and no increase in negative endpoint components relative to either comparator. Positivity-preserving decoder design is permissible but all transformations and corrections must be disclosed. Score the actual returned prediction; retain raw diagnostic values when a correction/projection is applied.
- End-to-end median inference latency <=1.5 times the fixed anchor on the declared hardware and batch, with startup/preparation reported separately. This is a proposed initial efficiency budget. A scientifically interesting slower method may be retained as TRADEOFF_ONLY, not quietly promoted.

All conditions are conjunctions, not terms in an agent-editable weighted score. A candidate that wins one metric but loses a guardrail is TRADEOFF_ONLY. A candidate with incomplete evidence is INCONCLUSIVE. Neither replaces the incumbent.

An optional ordering of already-admissible candidates is: maximize the minimum micro-acceptance gain across U/D and seeds; then lower worst supported group failure; then lower conditional tail error; then lower inference cost. Keep the full Pareto table even when one incumbent is selected.

## 8. Statistical interpretation and confirmation

Two paired seeds are a screen, not a confidence interval on all future training runs. A finalist should be checked with at least one additional prespecified seed; record the changed experimental role. Use paired state-level or source/block-level resampling if reporting uncertainty, retaining all species of a state together. Only four training snapshots cannot support a broad claim about new source realizations. Report per-source results and the limitation instead of treating hundreds of thousands of components as independent trials.

Require sealed confirmation to establish a new independently validated recipe. Repeatedly queried U/D results remain development findings. New domain/pressure/mechanism claims need their own reference data and frozen evaluation; one fixed-interval flame dataset is not a universal benchmark.

## 9. Scorer tests and anti-shortcuts

Required tests include: exact prediction passes; zero predictor fails nontrivial updates; wrong sign fails above the nontrivial floor; 1e-12 -> 1e-8 yields E=99000; tolerance boundaries are inclusive; empty groups are null; AR inclusion is explicit; row permutation is caught by the adapter; nonfinite predictions invalidate promotion; uncertainty masks are model-independent; replay uses no references; and repair/damage arithmetic is exact.

Maintain regression fixtures reproducing the historical MSE collapse and Smooth L1 transfer damage from saved records when the real arrays are available. A grader that calls those models improved is wrong. The worker may propose a grader fix as a separate change, but cannot grade its own candidate with an unapproved altered scorer.
