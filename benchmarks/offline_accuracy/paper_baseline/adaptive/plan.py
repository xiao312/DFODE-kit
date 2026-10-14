"""Fixed choices and immutable baseline bindings, declared before adaptive fitting."""
from ..matched_targets.plan import configuration as base_configuration
ARMS = ("continue", "local", "calibrated")
SEEDS = (20261011, 20261012)
CAMPAIGN_HASH = "f000d9b75a140c97cd5f11eea2365af003b89dda30aeed1dec576a09c6b8be24"
BASE_HASHES = {
    20261011: "80a8a534fe32592a1383b651d387322b40b02bfceb9005efafa47aa60a8eda9e",
    20261012: "99ed7295e586cd9800b60e1d56807fc22a508c4554ec6f5d18fbeafdd2552df2",
}


def configuration(arm, seed):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Select a declared adaptive arm and seed")
    return dict(experiment="residual-calibrated-local-v1", arm=arm, seed=seed,
        widths=[256]*3, updates=6000, batch_size=10000, normalization_rows=50000,
        diagnostics_every_updates=1000, learning_rate=1e-4, final_learning_rate=1e-6,
        wall_seconds=1200, scale_floor=1e-14, alpha_min=.01, alpha_max=100.,
        bin_edges=list(range(0,17,2)), minimum_bin_count=128, calibration_quantile=.75,
        checkpoint_selection="final", atol=1e-15, rtol=.1, device="cuda", interval=1e-6)


def base_config(seed):
    return base_configuration("gbct", "increment", seed)
