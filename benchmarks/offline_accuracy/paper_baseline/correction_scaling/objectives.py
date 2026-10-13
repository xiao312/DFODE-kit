"""New coordinate-only objective; historical objectives remain delegated unchanged."""
import torch
from ..resolved_fit.fit import objective as historical_objective


def objective(coordinate,base,truth,mask,target,config):
    if config["fit_objective"]=="coordinate-sl1":
        beta = config["coordinate_beta"]
        if not beta>0:
            raise ValueError("Coordinate beta must be positive")
        return torch.nn.functional.smooth_l1_loss(
            coordinate.double()[mask],target[mask],beta=beta,reduction="mean")
    return historical_objective(coordinate,base,truth,mask,target,config)
