"""Predeclared factorial comparison; do not select settings from development scores."""
from ..matched_work.plan import extended_configuration, batch_indices, update_policy as state_policy

TARGETS = ("state-boxcox", "signed-power", "scaled-asinh", "gbct")
OBJECTIVES = ("coordinate", "increment", "state")
SEEDS = (20261011, 20261012)


def configuration(target, objective, seed):
    if target not in TARGETS or objective not in OBJECTIVES or seed not in SEEDS:
        raise ValueError("Select a declared target, objective and seed")
    # Keep the state baseline's arithmetic schedule exactly, for every target.
    config = extended_configuration("fuel-state", seed)
    config.pop("recipe")
    config.update(experiment="matched-target-loss-200k-v1", target=target,
                  objective=objective, warmup=12000, atol=1e-15, rtol=.1,
                  asinh_scale=1e-14, target_normalization="mean-sample-std",
                  schedule="common-three-stage-adam-reset")
    return config


def update_policy(config, completed):
    return state_policy(dict(config, recipe="fuel-state"), completed)


def phase_objective(config, completed):
    if not 0 <= completed < config["updates"]:
        raise ValueError("Update outside declared plan")
    return "coordinate" if completed < config["warmup"] else config["objective"]
