"""Fixed diagnostic matrix, separate from frozen full-data campaigns."""
from ..decoder_loss.plan import configuration as decoder_configuration
from ..adaptive.plan import SEEDS

WIDTHS = (256, 512)
OBJECTIVES = ("physical", "direct")


def configuration(seed, width, objective):
    if seed not in SEEDS or width not in WIDTHS or objective not in OBJECTIVES:
        raise ValueError("Select a declared seed, width and objective")
    config = decoder_configuration("asinh-log", seed)
    config.update(experiment="resolved-training-fit-v1", widths=[width]*3,
                  fit_objective=objective, batch_size=16, wall_seconds=600,
                  diagnostics_every_updates=500, reference_fraction=.1,
                  feasibility_acceptance=.99)
    return config
