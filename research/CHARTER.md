# Research charter: magnitude and sign fidelity

## Objective

Develop and validate a generally applicable training and prediction recipe for signed chemical-kinetics increments. Improve physical increment acceptance across small and substantial targets, preserve already-correct predictions, and transfer the improvement beyond candidate-training states. Optimize the trained predictor and the evidence supporting it, not transformed training loss, the number of runs, or the volume of generated code.

At the current phase the prediction is a fixed-interval chemistry map:

    x = (T_initial, p_initial, Y_initial)
    d = Y_after_chemistry - Y_initial
    candidate(x) -> predicted signed d in mechanism species order

The inspected experiment is fixed-temperature/fixed-volume chemistry over 1e-6 s. Initial pressure is an input; this does not imply pressure is held constant during chemistry. Keep the mechanism, interval, reactor assumptions and data manifests pinned within a campaign. The phase concerns one-step increment fidelity, not an adiabatic reactor, a CVODE internal-error estimator, trajectory reliability or a CFD deployment claim.

## Immutable accuracy contract

For each active species component:

    tau(d) = 1e-15 + 0.1 * abs(d)
    E = abs(predicted_d - reference_d) / tau(reference_d)
    component_pass = E <= 1

Use the reference increment in the denominator. Never a predicted magnitude, train-fitted representation scale, or corrected tolerance. AR is excluded from the historical 58-species score; its behavior must still be validated by the inference adapter. Report complete-state acceptance as all 58 active species passing, not as a claim about every thermodynamic variable.

A zero prediction is acceptable when abs(d) <= 1e-15/(1-0.1). Above that magnitude, a zero or opposite-sign prediction cannot pass the contract. Small values below the absolute floor are not required to have uniformly accurate relative digits. Preserve reference uncertainty and do not claim digits not resolved by the reference calculation.

## Desired final property

One consistently applied inference rule should handle every covered state and species. Multiple internal modules, residual stages, learned routing and physical constraints may be investigated; they are not automatically excluded. However, do not use a manually tuned temperature switch, test-dependent rule, label lookup or hidden numerical integration to manufacture a gain. Any use of kinetic rates or numerical solves at inference changes the cost and method classification and must be disclosed and evaluated end to end.

The mission is architecture-neutral. GBCT, fixed asinh, the 3x256 correction head and hidden-feature access are hypotheses or controls, not permanent requirements. A method can be rejected without rejecting the research objective.

## Two axes of completion

An **experiment** is complete when its prespecified question was tested with valid, reproducible evidence and a documented conclusion. It may yield a negative, positive or inconclusive result.

A **candidate** is improved only when it passes the objective performance and preservation gates. A valid negative experiment does not promote its model. A passed unit test does not establish scientific improvement. An insightful explanation without reproducible evidence is a hypothesis.

## Milestones, not promises

M0: reproduce frozen baseline counts, identities and relevant numerical unit tests with the proposed evaluation adapter.

M1: demonstrate a reproducible incremental gain on both existing correction-unseen and development populations, in both designated seeds, subject to the guardrails in EVALUATION.md. This is a development-qualified research candidate, not independent confirmation.

M2: demonstrate at least five percentage points absolute acceptance gain over the frozen campaign anchor on both populations in both seeds, and at least 20% relative reduction in failure rate in the worst supported primary magnitude group, without a material regression in the other groups. Verify a frozen candidate on a new independently held case before calling the method independently confirmed. The numerical targets are proposed engineering milestones, not known attainable limits.

M3: an aspirational one-step target is at least 95% component acceptance overall and in each primary magnitude regime, with at least 99.9% sign accuracy on resolution-qualified nontrivial components, across prespecified independent cases. Report event counts and uncertainty; no finite test guarantees zero population failure. Even M3 is not a CFD/solver-reliability claim.

Do not silently turn an aspiration into a mandatory reason to spend unlimited compute. The owner approves finite research epochs, resource limits and the scope of final confirmation.

## Authority

The owner controls accuracy definitions, domain expansion, reference versions, permanent evaluation splits, resource budgets and deployment. A worker may propose changes to these items but cannot approve them while evaluating its own candidate. The worker may choose hypotheses, implement methods and execute the next justified bounded comparison inside an authorized campaign.
