#!/usr/bin/env python3
"""Diagnose physical delta quality and checkpoint/export equivalence."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import cantera as ct
import h5py
import numpy as np
import torch

from dfode_kit.training.positive_interval import PositiveIntervalTrainingConfig, _build_model
from scripts.export_fluent_artifact import FluentNeuralPatankarWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_prefix(path: str, limit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        pairs = handle["pairs"]
        count = min(limit, int(pairs["current_states"].shape[0]))
        current = np.asarray(pairs["current_states"][:count], dtype=np.float64)
        target = np.asarray(pairs["target_states"][:count], dtype=np.float64)
        dt = np.asarray(pairs["dt"][:count], dtype=np.float64).reshape(-1)
    return current, target, dt


def physical_metrics(predicted: np.ndarray, target: np.ndarray, current: np.ndarray) -> dict:
    true_delta = target[:, 2:] - current[:, 2:]
    error = predicted - true_delta
    true_abs = float(np.abs(true_delta).sum())
    dot = float(np.sum(predicted * true_delta))
    predicted_norm = float(np.linalg.norm(predicted))
    true_norm = float(np.linalg.norm(true_delta))
    row_true_norm = np.linalg.norm(true_delta, axis=1)
    row_predicted_norm = np.linalg.norm(predicted, axis=1)
    reactive = row_true_norm > 1.0e-12
    valid_cosine = reactive & (row_predicted_norm > 0.0)
    row_cosine = np.sum(predicted * true_delta, axis=1) / np.maximum(
        row_predicted_norm * row_true_norm, 1.0e-300
    )
    active_entries = np.abs(true_delta) > 1.0e-12
    return {
        "samples": int(current.shape[0]),
        "true_mean_abs_delta_y": float(np.abs(true_delta).mean()),
        "predicted_mean_abs_delta_y": float(np.abs(predicted).mean()),
        "predicted_max_abs_delta_y": float(np.abs(predicted).max()),
        "delta_mae": float(np.abs(error).mean()),
        "normalized_delta_mae": float(np.abs(error).sum() / max(true_abs, 1.0e-300)),
        "global_delta_cosine": dot / max(predicted_norm * true_norm, 1.0e-300),
        "median_reactive_row_cosine": float(np.median(row_cosine[valid_cosine])) if np.any(valid_cosine) else 0.0,
        "active_entry_sign_agreement": float(np.mean(np.sign(predicted[active_entries]) == np.sign(true_delta[active_entries]))) if np.any(active_entries) else 1.0,
        "negative_next_species_rate": float(np.mean(current[:, 2:] + predicted < 0.0)),
        "mean_mass_sum_error": float(np.mean(np.abs(np.sum(current[:, 2:] + predicted, axis=1) - 1.0))),
        "zero_baseline_normalized_delta_mae": 1.0,
    }


def wrapper_species_delta(value: torch.Tensor, species_count: int) -> torch.Tensor:
    """Extract delta_Y from a runtime wrapper with an optional delta_T column."""
    if value.ndim != 2:
        raise ValueError(f"runtime wrapper returned rank-{value.ndim} output")
    if value.shape[1] == species_count:
        return value
    if value.shape[1] == species_count + 1:
        return value[:, 1:]
    raise ValueError(
        "runtime wrapper output width must be n_species or n_species + 1; "
        f"received {value.shape[1]} for {species_count} species"
    )


def artifact_delta(module, physical_input, y0, direct_delta):
    attempts = []
    for label, arguments in (
        ("physical_input_y0", (physical_input, y0)),
        ("physical_input", (physical_input,)),
    ):
        try:
            value = module(*arguments)
        except Exception as error:
            attempts.append({"call": label, "error": str(error).splitlines()[0]})
            continue
        if isinstance(value, dict):
            candidates = list(value.values())
        elif isinstance(value, (tuple, list)):
            candidates = list(value)
        else:
            candidates = [value]
        for candidate in candidates:
            if not isinstance(candidate, torch.Tensor):
                continue
            array = candidate.detach().cpu().numpy()
            interpretations = []
            if array.shape == direct_delta.shape:
                interpretations.extend((("delta", array), ("next_species", array - y0.detach().cpu().numpy())))
            if array.ndim == 2 and array.shape[1] == direct_delta.shape[1] + 2:
                interpretations.append(("next_state", array[:, 2:] - y0.detach().cpu().numpy()))
            for interpretation, delta in interpretations:
                attempts.append(
                    {
                        "call": label,
                        "interpretation": interpretation,
                        "max_abs_delta_difference": float(np.max(np.abs(delta - direct_delta))),
                        "delta": delta,
                    }
                )
    valid = [item for item in attempts if "delta" in item]
    if not valid:
        return None, [{key: value for key, value in item.items() if key != "delta"} for item in attempts]
    best = min(valid, key=lambda item: item["max_abs_delta_difference"])
    diagnostics = [{key: value for key, value in item.items() if key != "delta"} for item in attempts]
    return best["delta"], diagnostics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    allowed = {field.name for field in fields(PositiveIntervalTrainingConfig)}
    config = PositiveIntervalTrainingConfig(
        **{key: value for key, value in checkpoint["training_config"].items() if key in allowed}
    )
    gas = ct.Solution(args.mechanism, checkpoint.get("phase_name"))
    model = _build_model(config, gas, len(checkpoint["state_mean"])).to(device)
    model.load_state_dict(checkpoint["net"])
    model.eval()
    eager_wrapper = FluentNeuralPatankarWrapper(
        model,
        checkpoint["state_mean"],
        checkpoint["state_std"],
        float(np.asarray(checkpoint["log_dt_mean"]).reshape(-1)[0]),
        float(np.asarray(checkpoint["log_dt_std"]).reshape(-1)[0]),
    ).to(device).eval()
    state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float64)
    state_std = np.asarray(checkpoint["state_std"], dtype=np.float64)
    log_dt_mean = np.asarray(checkpoint["log_dt_mean"], dtype=np.float64)
    log_dt_std = np.asarray(checkpoint["log_dt_std"], dtype=np.float64)
    artifact = torch.jit.load(args.artifact, map_location=device).eval() if args.artifact else None
    report = {
        "checkpoint": args.checkpoint,
        "artifact": args.artifact,
        "artifact_schema": str(artifact.forward.schema) if artifact is not None else None,
        "training_config": checkpoint["training_config"],
        "datasets": {},
    }
    for specification in args.dataset:
        name, path = specification.split("=", 1)
        current, target, dt = load_prefix(path, args.max_samples)
        x_np = ((current - state_mean) / state_std).astype(np.float32)
        t_np = ((np.log(np.maximum(dt, 1.0e-300))[:, None] - log_dt_mean) / log_dt_std).astype(np.float32)
        physical_np = np.concatenate([current, dt[:, None]], axis=1).astype(np.float32)
        runtime_x = (
            (
                physical_np[:, :-1].astype(np.float64)
                - state_mean
            )
            / state_std
        ).astype(np.float32)
        runtime_t = (
            np.log(np.maximum(physical_np[:, -1:], np.float32(1.0e-30)))
            - np.float32(log_dt_mean.reshape(-1)[0])
        ) / np.float32(log_dt_std.reshape(-1)[0])
        predicted_parts = []
        forward_parts = []
        reverse_parts = []
        availability_parts = []
        artifact_parts = []
        runtime_normalized_parts = []
        zero_time_parts = []
        eager_sparse_parts = []
        eager_dense_parts = []
        artifact_attempts = None
        component_audit = None
        with torch.inference_mode():
            for start in range(0, current.shape[0], args.batch_size):
                stop = min(start + args.batch_size, current.shape[0])
                x = torch.from_numpy(x_np[start:stop]).to(device)
                t = torch.from_numpy(t_np[start:stop]).to(device)
                physical = torch.from_numpy(physical_np[start:stop]).to(device)
                y0 = torch.from_numpy(current[start:stop, 2:]).to(device)
                output = model(x, t, current_species=y0)
                direct = output["delta_species"].cpu().numpy()
                predicted_parts.append(direct)
                runtime_output = model(
                    torch.from_numpy(runtime_x[start:stop]).to(device),
                    torch.from_numpy(runtime_t[start:stop]).to(device),
                    current_species=y0,
                )
                runtime_normalized_parts.append(runtime_output["delta_species"].cpu().numpy())
                zero_time_output = model(
                    x,
                    torch.zeros_like(t),
                    current_species=y0,
                )
                zero_time_parts.append(zero_time_output["delta_species"].cpu().numpy())
                eager_sparse_parts.append(
                    wrapper_species_delta(
                        eager_wrapper(physical, y0), y0.shape[1]
                    ).cpu().numpy()
                )
                eager_dense_parts.append(
                    wrapper_species_delta(
                        eager_wrapper.dense_reference(physical, y0), y0.shape[1]
                    ).cpu().numpy()
                )
                if component_audit is None:
                    runtime_x_tensor = torch.from_numpy(runtime_x[start:stop]).to(device)
                    runtime_t_tensor = torch.from_numpy(runtime_t[start:stop]).to(device)
                    wrapper_x, _wrapper_state_low = (
                        eager_wrapper._normalize_state(physical[:, :-1])
                    )
                    wrapper_t = (
                        torch.log(torch.clamp(physical[:, -1:], min=1.0e-30))
                        - eager_wrapper.log_dt_mean
                    ) / eager_wrapper.log_dt_std
                    direct_hidden = model.encoder(torch.cat([runtime_x_tensor, runtime_t_tensor], dim=-1))
                    wrapper_hidden = eager_wrapper.encoder(torch.cat([wrapper_x, wrapper_t], dim=-1))
                    direct_demand = torch.nn.functional.softplus(
                        model.demand_head(direct_hidden).to(torch.float64)
                    ) * model.extent_scale
                    wrapper_demand = torch.nn.functional.softplus(
                        eager_wrapper.demand_head(wrapper_hidden).to(torch.float64)
                    ) * eager_wrapper.extent_scale
                    component_audit = {
                        "normalized_state_max_abs_difference": float(torch.max(torch.abs(runtime_x_tensor - wrapper_x))),
                        "normalized_dt_max_abs_difference": float(torch.max(torch.abs(runtime_t_tensor - wrapper_t))),
                        "hidden_max_abs_difference": float(torch.max(torch.abs(direct_hidden - wrapper_hidden))),
                        "demand_max_abs_difference": float(torch.max(torch.abs(direct_demand - wrapper_demand))),
                        "consumption_max_abs_difference": float(torch.max(torch.abs(model.consumption - eager_wrapper.consumption))),
                        "process_stoich_max_abs_difference": float(torch.max(torch.abs(model.process_stoich - eager_wrapper.process_stoich))),
                        "model_extent_scale": float(model.extent_scale),
                        "wrapper_extent_scale": float(eager_wrapper.extent_scale),
                    }
                forward_parts.append(output["forward_extent"].cpu().numpy())
                reverse_parts.append(output["reverse_extent"].cpu().numpy())
                availability_parts.append(output["minimum_process_availability"].cpu().numpy())
                if artifact is not None:
                    artifact_prediction, attempts = artifact_delta(artifact, physical, y0, direct)
                    artifact_attempts = attempts
                    if artifact_prediction is not None:
                        artifact_parts.append(artifact_prediction)
        predicted = np.concatenate(predicted_parts)
        runtime_normalized_prediction = np.concatenate(runtime_normalized_parts)
        zero_time_prediction = np.concatenate(zero_time_parts)
        eager_sparse_prediction = np.concatenate(eager_sparse_parts)
        eager_dense_prediction = np.concatenate(eager_dense_parts)
        dataset_report = physical_metrics(predicted, target, current)
        dataset_report["component_audit"] = component_audit
        dataset_report["checkpoint_with_runtime_normalization"] = physical_metrics(
            runtime_normalized_prediction, target, current
        )
        dataset_report["checkpoint_with_zero_normalized_time"] = physical_metrics(
            zero_time_prediction, target, current
        )
        dataset_report["checkpoint_runtime_normalization_max_abs_delta_difference"] = float(
            np.max(np.abs(predicted - runtime_normalized_prediction))
        )
        dataset_report["eager_sparse_wrapper"] = physical_metrics(
            eager_sparse_prediction, target, current
        )
        dataset_report["eager_dense_wrapper"] = physical_metrics(
            eager_dense_prediction, target, current
        )
        dataset_report["runtime_checkpoint_eager_dense_max_abs_delta_difference"] = float(
            np.max(np.abs(runtime_normalized_prediction - eager_dense_prediction))
        )
        dataset_report["eager_sparse_dense_max_abs_delta_difference"] = float(
            np.max(np.abs(eager_sparse_prediction - eager_dense_prediction))
        )
        normalization_error = np.abs(runtime_x.astype(np.float64) - x_np.astype(np.float64))
        feature_names = ["T", "P", *checkpoint["species_names"]]
        ranked_features = np.argsort(np.nanmax(normalization_error, axis=0))[::-1]
        dataset_report["runtime_normalization_top_errors"] = [
            {
                "feature": str(feature_names[index]),
                "state_mean": float(state_mean[index]),
                "state_std": float(state_std[index]),
                "max_abs_normalized_error": float(np.nanmax(normalization_error[:, index])),
                "mean_abs_normalized_error": float(np.nanmean(normalization_error[:, index])),
            }
            for index in ranked_features[:8]
        ]
        forward = np.concatenate(forward_parts)
        reverse = np.concatenate(reverse_parts)
        availability = np.concatenate(availability_parts)
        dataset_report["mean_forward_extent"] = float(np.mean(forward))
        dataset_report["mean_reverse_extent"] = float(np.mean(reverse))
        dataset_report["simultaneous_forward_reverse_rate"] = float(np.mean((forward > 1.0e-16) & (reverse > 1.0e-16)))
        dataset_report["availability_limited_row_rate"] = float(np.mean(availability < 1.0 - 1.0e-10))
        if artifact_parts:
            artifact_prediction = np.concatenate(artifact_parts)
            dataset_report["artifact"] = physical_metrics(artifact_prediction, target, current)
            dataset_report["checkpoint_artifact_max_abs_delta_difference"] = float(np.max(np.abs(predicted - artifact_prediction)))
        if artifact_attempts is not None:
            dataset_report["artifact_call_diagnostics"] = artifact_attempts
        report["datasets"][name] = dataset_report
        print(json.dumps({"dataset": name, **dataset_report}, sort_keys=True), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
