#!/usr/bin/env python3
"""Replace a dense Fluent Patankar wrapper with a sparse equivalent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extent_scale_from_module(module: torch.jit.ScriptModule) -> float:
    value = getattr(module, "extent_scale", None)
    if value is not None:
        return float(value)
    _, constants = module.code_with_constants
    try:
        value = constants.c0
    except (AttributeError, KeyError) as exc:
        raise RuntimeError(
            "Could not recover Patankar extent scale"
        ) from exc
    if value is None or not isinstance(value, torch.Tensor):
        raise RuntimeError("Could not recover Patankar extent scale")
    if value.numel() != 1:
        raise RuntimeError("Patankar extent-scale constant is not scalar")
    return float(value.item())


class SparsePatankarWrapper(torch.nn.Module):
    def __init__(self, dense: torch.jit.ScriptModule):
        super().__init__()
        self.encoder = dense.encoder
        self.demand_head = dense.demand_head
        self.extent_scale = extent_scale_from_module(dense)
        self.register_buffer("consumption", dense.consumption.detach().clone())
        self.register_buffer(
            "process_stoich",
            dense.process_stoich.detach().clone(),
        )
        self.register_buffer("state_mean", dense.state_mean.detach().clone())
        self.register_buffer("state_std", dense.state_std.detach().clone())
        self.register_buffer(
            "log_dt_mean",
            dense.log_dt_mean.detach().clone(),
        )
        self.register_buffer(
            "log_dt_std",
            dense.log_dt_std.detach().clone(),
        )

        process_consumption = self.consumption.T
        reactant_counts = (process_consumption > 0.0).sum(dim=1)
        max_reactants = int(reactant_counts.max().item())
        reactant_indices = torch.zeros(
            (process_consumption.shape[0], max_reactants),
            dtype=torch.int64,
        )
        reactant_mask = torch.zeros_like(reactant_indices, dtype=torch.bool)
        for process_index in range(process_consumption.shape[0]):
            indices = torch.nonzero(
                process_consumption[process_index] > 0.0,
                as_tuple=False,
            ).flatten()
            reactant_indices[process_index, : indices.numel()] = indices
            reactant_mask[process_index, : indices.numel()] = True
        self.register_buffer("reactant_indices", reactant_indices)
        self.register_buffer("reactant_mask", reactant_mask)

    def forward(
        self,
        physical_input: torch.Tensor,
        current_species: torch.Tensor,
    ) -> torch.Tensor:
        state = physical_input[:, :-1]
        dt = torch.clamp(physical_input[:, -1:], min=1.0e-30)
        state_normalized = (state - self.state_mean) / self.state_std
        log_dt_normalized = (
            torch.log(dt) - self.log_dt_mean
        ) / self.log_dt_std
        hidden = self.encoder(
            torch.cat([state_normalized, log_dt_normalized], dim=-1)
        )
        demand = torch.nn.functional.softplus(
            self.demand_head(hidden).to(torch.float64)
        ) * self.extent_scale
        current_species = current_species.to(torch.float64)
        requested = demand @ self.consumption.T
        availability = torch.clamp(
            current_species / requested.clamp_min(1.0e-30),
            min=0.0,
            max=1.0,
        )
        reactant_availability = availability[:, self.reactant_indices]
        process_availability = torch.where(
            self.reactant_mask[None],
            reactant_availability,
            torch.ones(
                (),
                dtype=torch.float64,
                device=availability.device,
            ),
        ).amin(dim=-1)
        process_extent = (
            demand * process_availability * (1.0 - 1.0e-12)
        )
        return process_extent @ self.process_stoich.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    dense = torch.jit.load(str(args.input / "model.pt"), map_location="cpu")
    dense.eval()
    reference = np.load(args.input / "reference_io.npz")
    physical_input = torch.from_numpy(
        np.asarray(reference["physical_input"], dtype=np.float32)
    )
    current_species = torch.from_numpy(
        np.asarray(reference["current_species"], dtype=np.float64)
    )
    sparse = SparsePatankarWrapper(dense).eval()
    with torch.inference_mode():
        dense_output = dense(physical_input, current_species)
        sparse_output = sparse(physical_input, current_species)
        equivalence_error = float(
            torch.max(torch.abs(dense_output - sparse_output)).item()
        )
        traced = torch.jit.trace(
            sparse,
            (physical_input, current_species),
            strict=True,
        )
        trace_error = float(
            torch.max(
                torch.abs(
                    sparse_output
                    - traced(physical_input, current_species)
                )
            ).item()
        )
    if equivalence_error > 1.0e-14 or trace_error > 1.0e-14:
        raise RuntimeError(
            "Sparse artifact validation failed: "
            f"dense_error={equivalence_error}, trace_error={trace_error}"
        )

    shutil.copytree(args.input, args.output, dirs_exist_ok=True)
    model_path = args.output / "model.pt"
    traced.save(str(model_path))
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hard_layer"]["limiter_implementation"] = (
        "sparse-reactant-index"
    )
    manifest["sparse_equivalence_max_abs_error"] = equivalence_error
    manifest["reference_max_abs_error"] = trace_error
    manifest["checksums"]["model_sha256"] = sha256(model_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "extent_scale": sparse.extent_scale,
                "process_count": int(sparse.reactant_indices.shape[0]),
                "max_reactants": int(sparse.reactant_indices.shape[1]),
                "dense_equivalence_max_abs_error": equivalence_error,
                "trace_max_abs_error": trace_error,
                "model_sha256": manifest["checksums"]["model_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
