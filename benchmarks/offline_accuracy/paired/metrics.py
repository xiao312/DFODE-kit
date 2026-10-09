"""Same signed-increment error, two explicit allowance definitions."""
import numpy as np
from benchmarks.offline_accuracy.metrics import ABSOLUTE_FLOORS, RELATIVE_TOLERANCES, MAGNITUDE_EDGES
from .plan import POLICIES


def allowance(initial, reference, policy, atol, rtol):
    if policy not in POLICIES or not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0:
        raise ValueError("Invalid named error budget")
    magnitude = abs(reference) if policy == POLICIES[0] else abs(initial+reference)
    return atol + rtol*magnitude


def counts(error, budget, uncertainty=None):
    ratio = error/budget
    passing = error <= budget
    result = dict(components=int(error.size), states=len(error),
                  component_pass_count=int(passing.sum()), state_pass_count=int(passing.all(axis=1).sum()),
                  component_pass_fraction=float(passing.mean()), state_pass_fraction=float(passing.all(axis=1).mean()),
                  normalized_quantiles=dict(zip(("p50", "p95", "p99", "max"),
                      map(float, np.quantile(ratio, [.5, .95, .99, 1])))))
    if uncertainty is not None:
        known = np.isfinite(uncertainty) & (uncertainty <= .1*budget)
        passed = known & (error+uncertainty <= budget)
        result["qualification"] = dict(components=int(known.sum()), states=int(known.all(axis=1).sum()),
            passed_components=int(passed.sum()), passed_states=int(passed.all(axis=1).sum()),
            scope="Empirical uncertainty screen, not a rigorous bound")
    return result


def summarize(prediction, reference, initial, species, uncertainty=None):
    if (prediction.shape != reference.shape or initial.shape != reference.shape
            or reference.ndim != 2 or not len(reference) or len(species) != reference.shape[1]
            or not all(np.isfinite(x).all() for x in (prediction, reference, initial))):
        raise ValueError("Require aligned finite nonempty physical arrays")
    active = np.array([name != "AR" for name in species])
    prediction, reference, initial = (x[:, active] for x in (prediction, reference, initial))
    u = None if uncertainty is None else uncertainty[:, active]
    error = abs(prediction-reference)
    result = {}
    for policy in POLICIES:
        grid = []
        pairs = [(a, r, "paired-grid") for a in ABSOLUTE_FLOORS for r in RELATIVE_TOLERANCES]
        if policy == POLICIES[1]:
            pairs.append((1e-12, 1e-6, "application-state-diagnostic"))
        for atol, rtol, role in pairs:
            budget = allowance(initial, reference, policy, atol, rtol)
            grid.append(dict(atol=atol, rtol=rtol, role=role, **counts(error, budget, u)))
        budget = allowance(initial, reference, policy, 1e-15, .1)
        per_species = [dict(species=name, **counts(error[:, j:j+1], budget[:, j:j+1]))
                       for j, name in enumerate(np.array(species)[active])]
        bins = []
        for lower, upper in zip(MAGNITUDE_EDGES[:-1], MAGNITUDE_EDGES[1:]):
            selected = (abs(reference) >= lower) & (abs(reference) < upper)
            bins.append(dict(lower=lower, upper=None if np.isinf(upper) else upper,
                components=int(selected.sum()), component_pass_count=int(((error <= budget) & selected).sum())))
        result[policy] = dict(grid=grid, per_species=per_species, magnitude_bins=bins)
    return result
