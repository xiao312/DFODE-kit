"""Immutable incumbent identities and one-factor-at-a-time experimental choices."""
from ..adaptive.plan import SEEDS, BASE_HASHES
from ..resolved_fit.plan import configuration as probe_configuration
import math

CONTINUED_HASHES = {
    20261011:"ee6058566cb7fa8f3cef92e5e8a3dd71da2e80fb88af3e6ff8813aee7aa83ab7",
    20261012:"7984be372d4b092226ddd4bf9582904230d5082b44ad66c95e716e7eb7a62220",
}
SIZES = (10000,50000,200000)
OBJECTIVES = ("physical","direct")
RUN_OBJECTIVES = (*OBJECTIVES,"coordinate-sl1")


def configuration(seed, size, objective):
    if size not in SIZES or objective not in RUN_OBJECTIVES:
        raise ValueError("Select a declared size and objective")
    if objective=="coordinate-sl1" and size!=10000:
        raise ValueError("Coordinate Smooth L1 is restricted to the reviewed 10k diagnostic")
    cfg = probe_configuration(seed,256,"direct" if objective=="coordinate-sl1" else objective)
    cfg.update(experiment="continued-direct-scaling-v1",training_count=size,
               batch_size=10000,wall_seconds=1200,diagnostics_every_updates=1000)
    if objective=="coordinate-sl1":
        cfg.update(fit_objective=objective,coordinate_beta=math.log1p(.1))
    return cfg
