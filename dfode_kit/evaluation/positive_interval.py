from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from dfode_kit.data.interval_pairs import load_interval_pair_arrays, load_interval_thermo_arrays
from dfode_kit.evaluation.conservation import conservation_summary
from dfode_kit.evaluation.metrics import summarize_predictions
from dfode_kit.evaluation.stoich_interval import evaluate_stoich_interval_model
from dfode_kit.models.positive_interval import NeuralPatankarIntervalModel, ReactionTrajectoryFreeEnergyModel
from dfode_kit.physics.atom_conservation import reaction_affinity_over_rt, reaction_stoichiometry
from dfode_kit.physics.positive_integration import (
    augmented_mass_constraint_matrix,
    relative_positive_projection,
    sandu_composition_projection,
    sandu_reaction_space_projection,
)


def _load_positive_model(checkpoint, gas, input_dim, device):
    cfg = checkpoint["training_config"]
    stoich = reaction_stoichiometry(gas)
    mw = np.asarray(gas.molecular_weights, dtype=np.float64)
    if checkpoint["positive_model_type"] == "neural-patankar":
        model = NeuralPatankarIntervalModel(
            input_dim,
            mw[:, None] * stoich.reactants,
            mw[:, None] * stoich.products,
            stoich.reversible,
            latent_dim=int(cfg["latent_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            extent_scale=float(cfg["extent_scale"]),
            availability_floor=float(cfg["positivity_floor"]),
        )
    else:
        model = ReactionTrajectoryFreeEnergyModel(
            input_dim,
            stoich.net,
            mw,
            latent_dim=int(cfg["latent_dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            trajectory_scale=float(cfg["trajectory_scale"]),
            mobility_scale=float(cfg["mobility_scale"]),
            proximal_steps=int(cfg["proximal_steps"]),
            proximal_beta=float(cfg["proximal_beta"]),
            positivity_floor=float(cfg["positivity_floor"]),
        )
    model.load_state_dict(checkpoint["net"])
    return model.to(device).eval()


def _compute_affinity(current, gas):
    stoich = reaction_stoichiometry(gas)
    values = np.empty((current.shape[0], gas.n_reactions), dtype=np.float32)
    import cantera as ct

    for index, state in enumerate(current):
        gas.TPY = state[0], state[1], state[2:]
        values[index] = reaction_affinity_over_rt(
            stoich.net,
            gas.chemical_potentials / (ct.gas_constant * gas.T),
        )
    return values


def _predict_positive(checkpoint_path, source_path, gas, device):
    current, target, dt, dt_bin, dt_edges, species_names, attrs = load_interval_pair_arrays(source_path, dtype=np.float64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    log_mean = np.asarray(checkpoint["log_dt_mean"], dtype=np.float64)
    log_std = np.asarray(checkpoint["log_dt_std"], dtype=np.float64)
    x = ((current - state_mean) / np.where(state_std > 0, state_std, 1)).astype(np.float32)
    t = ((np.log(np.maximum(dt, 1e-300))[:, None] - log_mean) / np.where(log_std > 0, log_std, 1)).astype(np.float32)
    model = _load_positive_model(checkpoint, gas, current.shape[1], device)
    affinity = None
    if checkpoint["positive_model_type"] == "reaction-trajectory":
        try:
            affinity, _activity, _reversible = load_interval_thermo_arrays(source_path, dtype=np.float32)
        except (KeyError, OSError, ValueError):
            affinity = _compute_affinity(current, gas)
    chunks = []
    free_energy_chunks = []
    batch_size = 2048
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, current.shape[0], batch_size):
            stop = start + batch_size
            kwargs = {
                "current_species": torch.tensor(current[start:stop, 2:], dtype=torch.float64, device=device)
            }
            args = [
                torch.tensor(x[start:stop], dtype=torch.float32, device=device),
                torch.tensor(t[start:stop], dtype=torch.float32, device=device),
            ]
            if affinity is not None:
                args.append(torch.tensor(affinity[start:stop], dtype=torch.float32, device=device))
            output = model(*args, **kwargs)
            chunks.append(output["next_state"].detach().cpu().numpy())
            if "local_free_energy_delta" in output:
                free_energy_chunks.append(output["local_free_energy_delta"].detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    raw = np.concatenate(chunks)
    prediction = raw.copy()
    prediction[:, :2] = raw[:, :2] * state_std[:2] + state_mean[:2]
    thermo = None
    if free_energy_chunks:
        thermo = np.concatenate(free_energy_chunks).reshape(-1)
    return current, target, prediction, dt, dt_bin, dt_edges, species_names, attrs, runtime, thermo


def evaluate_positive_interval(
    checkpoint_path: str,
    source_path: str,
    mech_path: str,
    output_path: str,
    *,
    method: str = "none",
    device: str | None = None,
    floor: float = 1e-30,
) -> dict:
    import cantera as ct

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    gas = ct.Solution(mech_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    free_energy_delta = None
    if "positive_model_type" in checkpoint:
        current, target, prediction, dt, dt_bin, dt_edges, species_names, attrs, runtime, free_energy_delta = _predict_positive(
            checkpoint_path, source_path, gas, torch_device
        )
    else:
        base = evaluate_stoich_interval_model(
            checkpoint_path,
            source_path,
            mech_path=mech_path,
            device=str(torch_device),
            _return_prediction_arrays=True,
        )
        arrays = base["_prediction_arrays"]
        current, target, prediction = arrays["current"], arrays["target"], arrays["prediction"]
        dt, dt_bin, dt_edges = arrays["dt"], arrays["dt_bin"], arrays["dt_bin_edges"]
        species_names, attrs = arrays["species_names"], arrays["attrs"]
        runtime = base["neural_prediction_runtime"]["seconds"]

    projection_diagnostics = None
    projection_seconds = 0.0
    if method != "none":
        projection_started = time.perf_counter()
        y0 = torch.tensor(current[:, 2:], dtype=torch.float64, device=torch_device)
        yhat = torch.tensor(prediction[:, 2:], dtype=torch.float64, device=torch_device)
        constraint = torch.tensor(augmented_mass_constraint_matrix(gas), dtype=torch.float64, device=torch_device)
        if method == "sandu-composition":
            corrected, projection_diagnostics = sandu_composition_projection(yhat, y0, constraint, floor=floor)
        elif method == "relative":
            corrected, projection_diagnostics = relative_positive_projection(yhat, y0, constraint, floor=floor)
        elif method == "sandu-reaction":
            s_mass = torch.tensor(reaction_stoichiometry(gas).mass_fraction_matrix, dtype=torch.float64, device=torch_device)
            extent = (yhat - y0) @ torch.linalg.pinv(s_mass).T
            _extent, corrected, projection_diagnostics = sandu_reaction_space_projection(
                extent, y0, s_mass, floor=floor
            )
        else:
            raise ValueError(f"unsupported correction method: {method}")
        prediction[:, 2:] = corrected.detach().cpu().numpy()
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        projection_seconds = time.perf_counter() - projection_started

    species_true = target[:, 2:]
    species_pred = prediction[:, 2:]
    report = {
        "checkpoint": checkpoint_path,
        "source": source_path,
        "method": method,
        "n_pairs": int(current.shape[0]),
        "species_metrics": summarize_predictions(
            species_true,
            species_pred,
            species_names=species_names,
            small_thresholds=(1e-15, 1e-12),
        ),
        "temperature_mae": float(np.mean(np.abs(prediction[:, 0] - target[:, 0]))),
        "negative_entry_rate": float(np.mean(species_pred < 0.0)),
        "negative_row_rate": float(np.mean(np.any(species_pred < 0.0, axis=1))),
        "conservation_metrics": conservation_summary(gas, species_true, species_pred),
        "neural_runtime_seconds": float(runtime),
        "projection_runtime_seconds": float(projection_seconds),
    }
    if free_energy_delta is not None:
        report["local_free_energy_metrics"] = {
            "mean_delta": float(np.mean(free_energy_delta)),
            "max_delta": float(np.max(free_energy_delta)),
            "positive_delta_rate": float(np.mean(free_energy_delta > 1e-12)),
        }
    if projection_diagnostics is not None:
        report["projection_diagnostics"] = projection_diagnostics.__dict__
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
