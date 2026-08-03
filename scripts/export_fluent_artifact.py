#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from dfode_kit.evaluation.positive_interval import _load_positive_model
from dfode_kit.evaluation.stoich_interval import _load_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FluentReactionChannelWrapper(torch.nn.Module):
    """Map physical CFD inputs to transformed reaction channels."""

    def __init__(
        self,
        model: torch.nn.Module,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        log_dt_mean: float,
        log_dt_std: float,
    ):
        super().__init__()
        self.encoder = model.encoder
        self.flux_head = model.flux_head
        self.register_buffer(
            "state_mean",
            torch.as_tensor(state_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "state_std",
            torch.as_tensor(state_std, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_mean",
            torch.tensor(log_dt_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_std",
            torch.tensor(log_dt_std, dtype=torch.float32),
        )

    def forward(self, physical_input: torch.Tensor) -> torch.Tensor:
        state = physical_input[:, :-1]
        dt = torch.clamp(physical_input[:, -1:], min=1.0e-30)
        state_normalized = (state - self.state_mean) / self.state_std
        log_dt_normalized = (
            torch.log(dt) - self.log_dt_mean
        ) / self.log_dt_std
        latent = self.encoder(
            torch.cat([state_normalized, log_dt_normalized], dim=-1)
        )
        return self.flux_head(latent)


class FluentNeuralPatankarWrapper(torch.nn.Module):
    """Map physical CFD inputs directly to positive conservative delta_Y."""

    def __init__(
        self,
        model: torch.nn.Module,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        log_dt_mean: float,
        log_dt_std: float,
    ):
        super().__init__()
        self.encoder = model.encoder
        self.demand_head = model.demand_head
        self.extent_scale = float(model.extent_scale)
        self.register_buffer("consumption", model.consumption.detach().clone())
        self.register_buffer(
            "process_stoich",
            model.process_stoich.detach().clone(),
        )
        process_consumption = model.consumption.detach().clone().T
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
        self.register_buffer(
            "state_mean",
            torch.as_tensor(state_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "state_std",
            torch.as_tensor(state_std, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_mean",
            torch.tensor(log_dt_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "log_dt_std",
            torch.tensor(log_dt_std, dtype=torch.float32),
        )

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

    def dense_reference(
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
        process_availability = torch.where(
            self.consumption.T[None] > 0.0,
            availability[:, None, :],
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


def _reference_inputs(
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: float,
    log_dt_std: float,
) -> np.ndarray:
    offsets = np.asarray([-0.15, -0.05, 0.05, 0.15], dtype=np.float64)
    states = state_mean[None, :] + offsets[:, None] * state_std[None, :]
    dt = np.exp(log_dt_mean + offsets * log_dt_std)[:, None]
    return np.concatenate([states, dt], axis=1).astype(np.float32)


def export_artifact(
    checkpoint_path: Path,
    mechanism_path: Path,
    output_dir: Path,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_type = checkpoint.get("model_type", "stoichiometric_interval")
    positive_model_type = checkpoint.get("positive_model_type")
    is_neural_patankar = positive_model_type == "neural-patankar"
    if model_type != "stoichiometric_interval" and not is_neural_patankar:
        raise ValueError(
            "The Fluent exporter supports stoichiometric_interval and "
            f"neural-patankar checkpoints, got {model_type!r}/"
            f"{positive_model_type!r}"
        )

    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    log_dt_mean = float(np.asarray(checkpoint["log_dt_mean"]).reshape(-1)[0])
    log_dt_std = float(np.asarray(checkpoint["log_dt_std"]).reshape(-1)[0])
    if log_dt_std <= 0.0:
        log_dt_std = 1.0

    species_names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in checkpoint["species_names"]
    ]
    if is_neural_patankar:
        import cantera as ct

        gas = ct.Solution(str(mechanism_path))
        stoichiometry = np.asarray(
            checkpoint["molecular_weights"],
            dtype=np.float64,
        )[:, None] * np.asarray(
            checkpoint["stoichiometric_matrix"],
            dtype=np.float64,
        )
    else:
        stoichiometry = np.asarray(
            checkpoint["stoichiometric_mass_matrix"],
            dtype=np.float64,
            order="C",
        )
    n_species, n_reactions = stoichiometry.shape
    if state_mean.size != n_species + 2:
        raise ValueError("Checkpoint state and stoichiometry dimensions disagree")
    if len(species_names) != n_species:
        raise ValueError("Checkpoint species names and stoichiometry disagree")

    if is_neural_patankar:
        model = _load_positive_model(
            checkpoint,
            gas,
            state_mean.size,
            torch.device("cpu"),
        )
        wrapper = FluentNeuralPatankarWrapper(
            model,
            state_mean,
            state_std,
            log_dt_mean,
            log_dt_std,
        ).eval()
        output_mode = "direct_delta_y"
        runtime_reaction_width = n_species
    else:
        model = _load_model(
            checkpoint,
            state_mean.size,
            torch.device("cpu"),
        )
        wrapper = FluentReactionChannelWrapper(
            model,
            state_mean,
            state_std,
            log_dt_mean,
            log_dt_std,
        ).eval()
        output_mode = "reaction_channel"
        runtime_reaction_width = n_reactions

    reference_input = _reference_inputs(
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
    )
    reference_tensor = torch.from_numpy(reference_input)
    reference_species = torch.from_numpy(
        np.asarray(reference_input[:, 2:-1], dtype=np.float64)
    )
    with torch.inference_mode():
        reference_output = (
            wrapper(reference_tensor, reference_species).cpu().numpy()
            if is_neural_patankar
            else wrapper(reference_tensor).cpu().numpy()
        )
        sparse_equivalence_max_abs_error = (
            float(
                np.max(
                    np.abs(
                        reference_output
                        - wrapper.dense_reference(
                            reference_tensor,
                            reference_species,
                        ).cpu().numpy()
                    ),
                    initial=0.0,
                )
            )
            if is_neural_patankar
            else 0.0
        )
        trace_input = (
            (reference_tensor, reference_species)
            if is_neural_patankar
            else reference_tensor
        )
        traced = torch.jit.trace(wrapper, trace_input, strict=True)
        traced_output = (
            traced(reference_tensor, reference_species).cpu().numpy()
            if is_neural_patankar
            else traced(reference_tensor).cpu().numpy()
        )
    if sparse_equivalence_max_abs_error > 1.0e-14:
        raise RuntimeError(
            "Sparse Patankar limiter differs from dense reference: "
            f"max_abs_error={sparse_equivalence_max_abs_error}"
        )
    max_abs_error = float(
        np.max(np.abs(reference_output - traced_output), initial=0.0)
    )
    if max_abs_error > 1.0e-6:
        raise RuntimeError(
            f"TorchScript export mismatch: max_abs_error={max_abs_error}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    traced.save(str(model_path))
    mechanism_output = output_dir / "mechanism.yaml"
    shutil.copy2(mechanism_path, mechanism_output)
    np.savez(
        output_dir / "normalization.npz",
        state_mean=state_mean,
        state_std=state_std,
        log_dt_mean=np.asarray([log_dt_mean], dtype=np.float64),
        log_dt_std=np.asarray([log_dt_std], dtype=np.float64),
    )
    np.savez(
        output_dir / "stoichiometry.npz",
        mass_fraction_matrix=stoichiometry,
        species_names=np.asarray(species_names),
    )
    stoichiometry.tofile(output_dir / "stoichiometry.f64")
    (output_dir / "species_names.txt").write_text(
        "\n".join(species_names) + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "reference_io.npz",
        physical_input=reference_input,
        current_species=reference_species.numpy(),
        model_output=reference_output,
    )

    training_config = checkpoint.get("training_config", {})
    transform_alpha = float(training_config.get("transform_alpha", 0.1))
    scale_by_alpha = bool(
        training_config.get("transform_scale_by_alpha", True)
    )
    runtime_config = {
        "schema_version": 2 if is_neural_patankar else 1,
        "input_width": n_species + 3,
        "species_width": n_species,
        "reaction_width": runtime_reaction_width,
        "transform_alpha": transform_alpha,
        "transform_scale_by_alpha": int(scale_by_alpha),
        "output_mode": output_mode,
        "species_input_mode": (
            "separate_float64"
            if is_neural_patankar
            else "embedded_float32"
        ),
    }
    (output_dir / "runtime.cfg").write_text(
        "".join(f"{key}={value}\n" for key, value in runtime_config.items()),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 2 if is_neural_patankar else 1,
        "artifact_type": (
            "dfode-fluent-direct-delta-y"
            if is_neural_patankar
            else "dfode-fluent-reaction-channel"
        ),
        "model_type": positive_model_type or model_type,
        "checkpoint": str(checkpoint_path),
        "mechanism": mechanism_output.name,
        "phase_name": checkpoint.get("phase_name"),
        "input": {
            "dtype": "float32",
            "shape": ["batch", n_species + 3],
            "fields": ["T", "P", *species_names, "dt"],
            "units": {
                "T": "K",
                "P": "Pa",
                "Y": "dimensionless",
                "dt": "s",
            },
        },
        "hard_layer_input": (
            {
                "dtype": "float64",
                "shape": ["batch", n_species],
                "fields": species_names,
                "meaning": (
                    "exact physical mass fractions used for "
                    "Patankar availability"
                ),
            }
            if is_neural_patankar
            else None
        ),
        "output": {
            "dtype": "float64" if is_neural_patankar else "float32",
            "shape": [
                "batch",
                n_species if is_neural_patankar else n_reactions,
            ],
            "meaning": (
                "positive stoichiometrically conservative delta_Y"
                if is_neural_patankar
                else "signed-power-transformed integrated reaction extents"
            ),
        },
        "hard_layer": {
            "equation": (
                "delta_Y = PatankarResourceAllocation(Y, demand)"
                if is_neural_patankar
                else "delta_Y = (W*S) @ reaction_extent"
            ),
            "dtype": "float64",
            "current_species_dtype": (
                "float64" if is_neural_patankar else None
            ),
            "current_species_source": (
                "separate_input" if is_neural_patankar else None
            ),
            "transform_alpha": transform_alpha,
            "transform_scale_by_alpha": scale_by_alpha,
            "limiter_implementation": (
                "sparse-reactant-index"
                if is_neural_patankar
                else None
            ),
        },
        "species_names": species_names,
        "n_reactions": n_reactions,
        "checksums": {
            "model_sha256": _sha256(model_path),
            "mechanism_sha256": _sha256(mechanism_output),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "reference_max_abs_error": max_abs_error,
        "sparse_equivalence_max_abs_error": (
            sparse_equivalence_max_abs_error
        ),
        "training_config": training_config,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a DFODE interval checkpoint for native Fluent inference."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mechanism", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = export_artifact(
        args.checkpoint.resolve(),
        args.mechanism.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
