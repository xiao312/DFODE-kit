#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

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
    if model_type != "stoichiometric_interval":
        raise ValueError(
            "The initial Fluent exporter supports model_type="
            f"'stoichiometric_interval', got {model_type!r}"
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

    model = _load_model(checkpoint, state_mean.size, torch.device("cpu"))
    wrapper = FluentReactionChannelWrapper(
        model,
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
    ).eval()

    reference_input = _reference_inputs(
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
    )
    reference_tensor = torch.from_numpy(reference_input)
    with torch.inference_mode():
        reference_output = wrapper(reference_tensor).cpu().numpy()
        traced = torch.jit.trace(wrapper, reference_tensor, strict=True)
        traced_output = traced(reference_tensor).cpu().numpy()
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
        transformed_reaction_channels=reference_output,
    )

    training_config = checkpoint.get("training_config", {})
    transform_alpha = float(training_config.get("transform_alpha", 0.1))
    scale_by_alpha = bool(
        training_config.get("transform_scale_by_alpha", True)
    )
    runtime_config = {
        "schema_version": 1,
        "input_width": n_species + 3,
        "species_width": n_species,
        "reaction_width": n_reactions,
        "transform_alpha": transform_alpha,
        "transform_scale_by_alpha": int(scale_by_alpha),
    }
    (output_dir / "runtime.cfg").write_text(
        "".join(f"{key}={value}\n" for key, value in runtime_config.items()),
        encoding="ascii",
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "dfode-fluent-reaction-channel",
        "model_type": model_type,
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
        "output": {
            "dtype": "float32",
            "shape": ["batch", n_reactions],
            "meaning": "signed-power-transformed integrated reaction extents",
        },
        "hard_layer": {
            "equation": "delta_Y = (W*S) @ reaction_extent",
            "dtype": "float64",
            "transform_alpha": transform_alpha,
            "transform_scale_by_alpha": scale_by_alpha,
        },
        "species_names": species_names,
        "n_reactions": n_reactions,
        "checksums": {
            "model_sha256": _sha256(model_path),
            "mechanism_sha256": _sha256(mechanism_output),
            "checkpoint_sha256": _sha256(checkpoint_path),
        },
        "reference_max_abs_error": max_abs_error,
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
