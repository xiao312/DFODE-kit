"""Immutable incumbent identities and one-factor-at-a-time experimental choices."""
from ..adaptive.plan import SEEDS, BASE_HASHES
from ..resolved_fit.plan import configuration as probe_configuration

CONTINUED_HASHES = {
    20261011:"ee6058566cb7fa8f3cef92e5e8a3dd71da2e80fb88af3e6ff8813aee7aa83ab7",
    20261012:"7984be372d4b092226ddd4bf9582904230d5082b44ad66c95e716e7eb7a62220",
}
SIZES = (10000,50000,200000)
OBJECTIVES = ("physical","direct")


def configuration(seed, size, objective):
    if size not in SIZES or objective not in OBJECTIVES:
        raise ValueError("Select a declared size and objective")
    cfg = probe_configuration(seed,256,objective)
    cfg.update(experiment="continued-direct-scaling-v1",training_count=size,
               batch_size=10000,wall_seconds=1200,diagnostics_every_updates=1000)
    return cfg
