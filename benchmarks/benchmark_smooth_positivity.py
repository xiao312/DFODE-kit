#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch

from dfode_kit.physics.smooth_positivity import (
    compare_hard_and_smooth_scaling,
    directional_smoothness_diagnostics,
    hard_reaction_extent_scaling,
    patankar_directional_smoothness_diagnostics,
    smooth_safe_reaction_extent_scaling,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _balanced_matrix(n_species, n_reactions, generator, device):
    matrix = torch.randn(
        n_species, n_reactions, generator=generator, dtype=torch.float64
    )
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    matrix = matrix / torch.linalg.vector_norm(
        matrix, dim=0, keepdim=True
    ).clamp_min(1e-30)
    return matrix.to(device)


def _batch(batch_size, n_species, n_reactions, generator, device):
    current = torch.pow(
        10.0,
        -20.0
        * torch.rand(
            batch_size, n_species, generator=generator, dtype=torch.float64
        ),
    )
    current = current / current.sum(dim=-1, keepdim=True)
    extent = torch.randn(
        batch_size, n_reactions, generator=generator, dtype=torch.float64
    )
    return current.to(device), extent.to(device)


def _scale_proposals(current, extent, stoich, generator):
    delta = extent @ stoich.T
    maximum_ratio = ((-delta).clamp_min(0.0) / (current + 1e-15)).amax(
        dim=-1, keepdim=True
    )
    target_ratio = torch.pow(
        10.0,
        -1.0
        + 2.0
        * torch.rand(
            current.shape[0], 1, generator=generator, dtype=torch.float64
        ).to(current.device),
    )
    return extent * target_ratio / maximum_ratio.clamp_min(1e-30)


def _composition_direction(current, generator):
    direction = torch.randn(
        current.shape, generator=generator, dtype=torch.float64
    ).to(current.device) * current
    direction = direction - current * direction.sum(dim=-1, keepdim=True)
    relative = torch.abs(direction) / current.clamp_min(1e-30)
    return direction / relative.amax(dim=-1, keepdim=True).clamp_min(1e-30)


def _time(function, repeats, device):
    for _ in range(5):
        function()
    _sync(device)
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    _sync(device)
    return (time.perf_counter() - started) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--species", type=int, default=53)
    parser.add_argument("--reactions", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--p-norm", type=float, default=32.0)
    parser.add_argument("--transition-width", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    stoich = _balanced_matrix(args.species, args.reactions, generator, device)
    current, extent = _batch(
        args.batch_size, args.species, args.reactions, generator, device
    )
    extent = _scale_proposals(current, extent, stoich, generator)
    hard, smooth, comparison = compare_hard_and_smooth_scaling(
        current,
        extent,
        stoich,
        numerical_tolerance=0.0,
        p_norm=args.p_norm,
        transition_width=args.transition_width,
    )

    subset = min(512, args.batch_size)
    directional = directional_smoothness_diagnostics(
        current[:subset],
        extent[:subset],
        stoich,
        _composition_direction(current[:subset], generator),
        epsilon=1e-6,
        p_norm=args.p_norm,
        transition_width=args.transition_width,
    )
    tie_stoich = torch.tensor(
        [[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
        device=device,
    )
    tie_current = torch.tensor(
        [[0.1, 0.1, 0.8]], dtype=torch.float64, device=device
    ).repeat(256, 1)
    tie_extent = torch.tensor(
        [[0.2, 0.2]], dtype=torch.float64, device=device
    ).repeat(256, 1)
    tie_direction = torch.tensor(
        [[1.0, -1.0, 0.0]], dtype=torch.float64, device=device
    ).repeat(256, 1)
    switching_stress = directional_smoothness_diagnostics(
        tie_current,
        tie_extent,
        tie_stoich,
        tie_direction,
        epsilon=1e-7,
        p_norm=args.p_norm,
        transition_width=args.transition_width,
    )
    patankar_stress = patankar_directional_smoothness_diagnostics(
        tie_current,
        torch.ones((256, 1), dtype=torch.float64, device=device),
        torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64, device=device),
        tie_direction,
        epsilon=1e-7,
        p_norm=args.p_norm,
    )

    hard_seconds = _time(
        lambda: hard_reaction_extent_scaling(current, extent, stoich),
        args.repeats,
        device,
    )
    smooth_seconds = _time(
        lambda: smooth_safe_reaction_extent_scaling(
            current,
            extent,
            stoich,
            numerical_tolerance=0.0,
            p_norm=args.p_norm,
            transition_width=args.transition_width,
        ),
        args.repeats,
        device,
    )
    report = {
        "configuration": vars(args),
        "representative_multiscale": comparison,
        "representative_directional": directional,
        "competing_species_stress": switching_stress,
        "patankar_reactant_stress": patankar_stress,
        "runtime": {
            "hard_microseconds_per_sample": 1e6 * hard_seconds / args.batch_size,
            "smooth_microseconds_per_sample": 1e6 * smooth_seconds / args.batch_size,
            "smooth_to_hard_runtime_ratio": smooth_seconds / hard_seconds,
        },
        "checks": {
            "smooth_scale_never_exceeds_hard": bool(
                torch.all(smooth.scale <= hard.scale + 2e-15)
            ),
            "smooth_minimum_next_y": float(smooth.next_y.min().detach().cpu()),
            "maximum_mass_sum_delta": float(
                torch.abs(smooth.delta_y.sum(dim=-1)).max().detach().cpu()
            ),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
