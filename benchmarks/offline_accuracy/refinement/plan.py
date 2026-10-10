"""Frozen comparison matrix; no result-dependent selection."""
from benchmarks.flame_conditioning.coordinates import KINDS

ATOL = 1e-15
RTOL = .1
COORDINATE_SCALE = 100.


def variants():
    common = dict(widths=[800] * 4, physical_weight=0., residual=False)
    result = [dict(common, name=f"long-{kind}", target=kind, updates=4000) for kind in KINDS]
    for kind in ("budget-log", "budget-asinh"):
        result.append(dict(common, name=kind, target=kind, updates=2000))
        result.append(dict(common, name=f"physical-{kind}", target=kind, updates=2000, physical_weight=.01))
    result.append(dict(common, name="residual-state-boxcox", target="residual", updates=2000, residual=True))
    result.append(dict(common, name="deep-state-boxcox", target="state-boxcox", updates=2000, widths=[800] * 8))
    return result


def configuration(name, seed):
    if seed not in (20261011, 20261012):
        raise ValueError("Use one of the two predeclared seeds")
    matches = [row for row in variants() if row["name"] == name]
    if len(matches) != 1:
        raise ValueError("Unknown variant; inspect --dry-run for the fixed matrix")
    return dict(matches[0], seed=seed, training_count=10000, batch_size=256,
                learning_rate=.001, final_learning_rate=.00001,
                validation_every=500, wall_seconds=900, atol=ATOL, rtol=RTOL)
