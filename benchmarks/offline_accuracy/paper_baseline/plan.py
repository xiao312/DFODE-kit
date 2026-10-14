"""Frozen reduced-data recipes; no outcome-dependent hyperparameters."""
def configuration(recipe, seed):
    if recipe not in ("fuel-state", "fuel-power") or seed not in (20261011, 20261012):
        raise ValueError("Select fuel-state/fuel-power and a declared seed")
    return dict(recipe=recipe, seed=seed, device="cuda", widths=[800]*4,
                epochs=1500 if recipe == "fuel-state" else 2000,
                batch_size=20000, interval=1e-6, wall_seconds=1200,
                checkpoint_selection="final", learning_rate=1e-3,
                decay_every=500, decay_factor=.1, diagnostics_every=100)


def epoch_policy(config, epoch):
    """Epoch is zero-indexed; boundaries match saved source scripts."""
    if not 0 <= epoch < config["epochs"]:
        raise ValueError("Epoch outside plan")
    # The state script resets AFTER zero-index epochs 500 and 1000.
    state = config["recipe"] == "fuel-state"
    stage = max(0, (epoch-1)//config["decay_every"]) if state else epoch//config["decay_every"]
    reset = epoch == 0 or (state and epoch > 1 and (epoch-1) % config["decay_every"] == 0)
    return config["learning_rate"]*config["decay_factor"]**stage, reset
