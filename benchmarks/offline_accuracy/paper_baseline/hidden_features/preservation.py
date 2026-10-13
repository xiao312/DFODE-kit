"""One declared training-only interval constraint, not an inference-time gate."""
import torch
from ..correction_scaling.objectives import objective as coordinate_objective


def objective(coordinate, base, truth, mask, target, config):
    ordinary = coordinate_objective(coordinate, base, truth, mask, target, config)
    allowance = 1e-15 + .1 * truth.abs()
    passing = ((base-truth).abs() <= allowance) & mask
    if not passing.any():
        raise ValueError("Preservation population is empty")
    center = torch.asinh(base / config["scale_floor"])
    lower = torch.asinh((truth-allowance) / config["scale_floor"]) - center
    upper = torch.asinh((truth+allowance) / config["scale_floor"]) - center
    distance = torch.relu(lower-coordinate.double()) + torch.relu(coordinate.double()-upper)
    penalty = (distance[passing] / config["coordinate_beta"]).square().mean()
    return ordinary + config["preservation_weight"] * penalty
