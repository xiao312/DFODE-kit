"""Predeclared work budget and update-indexed schedules."""
from ..plan import configuration as source_configuration


def configuration(recipe, seed):
    config = source_configuration(recipe, seed)
    for key in ("epochs", "decay_every", "diagnostics_every"):
        config.pop(key)
    config.update(experiment="fuel-matched-work-v1", updates=6000, batch_size=10000,
                  diagnostics_every_updates=1000, normalization_rows=50000,
                  schedule_unit="updates", normalization_policy="frozen-selected-50k")
    return config


def extended_configuration(recipe, seed):
    return dict(configuration(recipe, seed), experiment="fuel-fixed-data-budget-v1", updates=18000)


def update_policy(config, completed):
    if not 0 <= completed < config["updates"]:
        raise ValueError("Update outside declared plan")
    stages = 3 if config["recipe"] == "fuel-state" else 4
    if config["updates"] % stages:
        raise ValueError("Update budget must divide evenly into rate stages")
    length = config["updates"]//stages
    stage = completed//length
    reset = completed == 0 or (config["recipe"] == "fuel-state" and completed % length == 0)
    return config["learning_rate"]*config["decay_factor"]**stage, reset


def validate_pool(count, config):
    if count < config["batch_size"] or count % config["batch_size"]:
        raise ValueError("Pool must be a positive multiple of the fixed batch size")


def batch_indices(count, config):
    """No partial batches; return row positions, independent of targets or scores."""
    import numpy as np
    validate_pool(count, config)
    rng = np.random.default_rng(config["seed"])
    completed = 0
    while completed < config["updates"]:
        permutation = rng.permutation(count)
        for start in range(0, count, config["batch_size"]):
            if completed >= config["updates"]:
                return
            yield permutation[start:start+config["batch_size"]]
            completed += 1
