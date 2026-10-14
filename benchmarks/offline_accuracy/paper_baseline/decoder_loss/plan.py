"""Predeclared choices; no tuning against the development outcomes."""
from ..adaptive.plan import configuration as adaptive_configuration,SEEDS,BASE_HASHES,CAMPAIGN_HASH,base_config
ARMS = ("linear-log","linear-huber","asinh-log","asinh-huber")


def configuration(arm,seed):
    if arm not in ARMS:
        raise ValueError("Unknown decoder/loss arm")
    decoder,loss = arm.split("-")
    config = adaptive_configuration("local",seed)
    config.update(experiment="correction-decoder-loss-v1",arm=arm,decoder=decoder,objective=loss,
                  coordinate_offset=0.,coordinate_divisor=1.,gradient_clipping=None)
    return config
