#!/usr/bin/env python3
"""Evaluate DFODE directional sensitivities against Fluent-native DI pairs.

The input must use the ``fluent-di-jvp-v1`` schema. Labels from Cantera,
CVODE, or SUNDIALS are rejected by the schema loader. Each base and perturbed
state is independently passed through the exported TorchScript artifact and
the deployment hard limiter before finite-difference JVPs are formed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dfode_kit.data.cfd_conditioned import canonical_species_name
from dfode_kit.data.fluent_sensitivity import load_fluent_di_jvp_dataset


SCHEMA = "dfode.fluent_jvp_evaluation"
DEFAULT_LIMITER_SAFETY = 0.999999


def _model_call(
    model: torch.jit.ScriptModule,
    physical_input: torch.Tensor,
    current_species: torch.Tensor,
    separate_species_input: bool,
) -> torch.Tensor:
    result = (
        model(physical_input, current_species)
        if separate_species_input
        else model(physical_input)
    )
    if isinstance(result, (tuple, list)):
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("TorchScript artifact must return a tensor or tensor tuple")
    return result


def infer_artifact(
    artifact_dir: Path,
    states: np.ndarray,
    dt: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text("utf-8"))
    model = torch.jit.load(str(artifact_dir / "model.pt"), map_location=device).eval()
    separate_species_input = (
        manifest.get("hard_layer", {}).get("current_species_source")
        == "separate_input"
    )
    synchronize = str(device).startswith("cuda") and torch.cuda.is_available()

    warm_end = min(batch_size, states.shape[0])
    warm_input = np.concatenate(
        (states[:warm_end], dt[:warm_end, None]), axis=1
    ).astype(np.float32, copy=False)
    with torch.inference_mode():
        _model_call(
            model,
            torch.from_numpy(warm_input).to(device),
            torch.from_numpy(
                np.ascontiguousarray(states[:warm_end, 2:], dtype=np.float64)
            ).to(device),
            separate_species_input,
        )
    if synchronize:
        torch.cuda.synchronize()

    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, states.shape[0], batch_size):
            end = min(begin + batch_size, states.shape[0])
            physical = np.concatenate(
                (states[begin:end], dt[begin:end, None]), axis=1
            ).astype(np.float32, copy=False)
            result = _model_call(
                model,
                torch.from_numpy(physical).to(device),
                torch.from_numpy(
                    np.ascontiguousarray(states[begin:end, 2:], dtype=np.float64)
                ).to(device),
                separate_species_input,
            )
            outputs.append(result.detach().cpu().numpy().astype(np.float64, copy=False))
    if synchronize:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = np.concatenate(outputs, axis=0)
    return output, manifest, {
        "device": str(device),
        "batch_size": int(batch_size),
        "sample_count": int(states.shape[0]),
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(states.shape[0] / max(elapsed, 1.0e-300)),
        "microseconds_per_sample": float(1.0e6 * elapsed / states.shape[0]),
        "warmup_excluded": True,
    }


def split_artifact_output(
    output: np.ndarray, n_species: int
) -> tuple[np.ndarray | None, np.ndarray]:
    if output.ndim != 2:
        raise ValueError("artifact output must have shape (batch, output_width)")
    if output.shape[1] == n_species + 1:
        return output[:, 0], output[:, 1:]
    if output.shape[1] == n_species:
        return None, output
    raise ValueError(
        f"artifact output width {output.shape[1]} is incompatible with "
        f"{n_species} species"
    )


def apply_deployment_hard_limiter(
    current_species: np.ndarray,
    delta_y: np.ndarray,
    *,
    safety: float = DEFAULT_LIMITER_SAFETY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.asarray(current_species, dtype=np.float64)
    delta = np.asarray(delta_y, dtype=np.float64)
    if current.shape != delta.shape or current.ndim != 2:
        raise ValueError("current_species and delta_y must have equal 2D shapes")
    if not 0.0 < safety <= 1.0:
        raise ValueError("limiter safety must be in (0, 1]")
    candidates = np.full(delta.shape, np.inf, dtype=np.float64)
    consuming = delta < 0.0
    candidates[consuming] = safety * current[consuming] / (-delta[consuming])
    controlling = np.argmin(candidates, axis=1)
    minimum = candidates[np.arange(delta.shape[0]), controlling]
    scales = np.minimum(1.0, np.maximum(0.0, minimum))
    active = scales < 1.0
    controlling = np.where(active, controlling, -1).astype(np.int64, copy=False)
    return delta * scales[:, None], scales, controlling


def _endpoint(
    current: np.ndarray,
    delta_temperature: np.ndarray | None,
    delta_y: np.ndarray,
) -> np.ndarray:
    endpoint = np.asarray(current, dtype=np.float64).copy()
    if delta_temperature is not None:
        endpoint[:, 0] += delta_temperature
    endpoint[:, 2:] += delta_y
    return endpoint


def _finite_difference(after: np.ndarray, before: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    return (np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)) / epsilon[:, None]


def _regression_summary(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    true = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError("target and prediction shapes differ")
    error = pred - true
    target_l1 = float(np.sum(np.abs(true), dtype=np.float64))
    denominator = float(np.linalg.norm(true.reshape(-1)) * np.linalg.norm(pred.reshape(-1)))
    return {
        "sample_count": int(true.shape[0]),
        "entry_count": int(true.size),
        "mae": float(np.mean(np.abs(error))) if true.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(error * error))) if true.size else float("nan"),
        "nmae": (
            float(np.sum(np.abs(error), dtype=np.float64) / target_l1)
            if target_l1 > 0.0
            else float("nan")
        ),
        "cosine": (
            float(np.dot(true.reshape(-1), pred.reshape(-1)) / denominator)
            if denominator > 0.0
            else float("nan")
        ),
    }


def _signed_direction_agreement(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    zero_tolerance: float,
) -> dict[str, Any]:
    true = np.asarray(target, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    significant = np.abs(true) > zero_tolerance
    significant_count = int(np.count_nonzero(significant))
    entry_agreement = (
        float(np.mean(np.sign(true[significant]) == np.sign(pred[significant])))
        if significant_count
        else float("nan")
    )
    true_rows = true.reshape(true.shape[0], -1)
    pred_rows = pred.reshape(pred.shape[0], -1)
    true_norm = np.linalg.norm(true_rows, axis=1)
    valid_rows = true_norm > zero_tolerance
    dots = np.sum(true_rows * pred_rows, axis=1, dtype=np.float64)
    return {
        "zero_tolerance": float(zero_tolerance),
        "significant_entry_count": significant_count,
        "significant_entry_fraction": float(np.mean(significant)),
        "entrywise_sign_agreement": entry_agreement,
        "nonzero_target_sample_count": int(np.count_nonzero(valid_rows)),
        "positive_vector_alignment_fraction": (
            float(np.mean(dots[valid_rows] > 0.0))
            if np.any(valid_rows)
            else float("nan")
        ),
    }


def _species_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    species_names: Sequence[str],
    *,
    sign_zero_tolerance: float,
) -> dict[str, Any]:
    return {
        "overall": _regression_summary(target, prediction),
        "signed_direction_agreement": _signed_direction_agreement(
            target, prediction, zero_tolerance=sign_zero_tolerance
        ),
        "per_species": [
            {
                "species": str(name),
                **_regression_summary(target[:, index], prediction[:, index]),
                "signed_direction_agreement": _signed_direction_agreement(
                    target[:, index : index + 1],
                    prediction[:, index : index + 1],
                    zero_tolerance=sign_zero_tolerance,
                ),
            }
            for index, name in enumerate(species_names)
        ],
    }


def _paired_metrics(
    endpoint_target: np.ndarray,
    endpoint_prediction: np.ndarray,
    source_target: np.ndarray,
    source_prediction: np.ndarray,
    species_names: Sequence[str],
    *,
    predicts_temperature: bool,
    sign_zero_tolerance: float,
) -> dict[str, Any]:
    endpoint: dict[str, Any] = {
        "species": _species_summary(
            endpoint_target[:, 2:],
            endpoint_prediction[:, 2:],
            species_names,
            sign_zero_tolerance=sign_zero_tolerance,
        ),
        "pressure": _regression_summary(
            endpoint_target[:, 1], endpoint_prediction[:, 1]
        ),
    }
    if predicts_temperature:
        endpoint["full_state"] = _regression_summary(
            endpoint_target, endpoint_prediction
        )
        endpoint["temperature"] = _regression_summary(
            endpoint_target[:, 0], endpoint_prediction[:, 0]
        )
    else:
        unavailable = {
            "available": False,
            "reason": (
                "artifact predicts species only; Fluent-owned thermochemical "
                "temperature closure is not reproduced offline"
            ),
        }
        endpoint["full_state"] = unavailable
        endpoint["temperature"] = unavailable
    return {
        "endpoint_jvp": endpoint,
        "delta_y_over_dt_source_jvp": {
            "units": "mass-fraction/s per perturbation parameter",
            **_species_summary(
                source_target,
                source_prediction,
                species_names,
                sign_zero_tolerance=sign_zero_tolerance,
            ),
        },
    }


def _subset_metrics(
    mask: np.ndarray,
    endpoint_target: np.ndarray,
    endpoint_prediction: np.ndarray,
    source_target: np.ndarray,
    source_prediction: np.ndarray,
    species_names: Sequence[str],
    *,
    predicts_temperature: bool,
    sign_zero_tolerance: float,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(selected):
        return {"sample_count": 0}
    result = _paired_metrics(
        endpoint_target[selected],
        endpoint_prediction[selected],
        source_target[selected],
        source_prediction[selected],
        species_names,
        predicts_temperature=predicts_temperature,
        sign_zero_tolerance=sign_zero_tolerance,
    )
    result["sample_count"] = int(np.count_nonzero(selected))
    return result


def _categorical_bins(
    values: np.ndarray,
    metric_builder,
) -> list[dict[str, Any]]:
    labels = np.asarray(values).astype(str)
    rows = []
    for value in sorted(set(labels.tolist())):
        rows.append({"value": value, **metric_builder(labels == value)})
    return rows


def _quantile_edges(values: np.ndarray, count: int) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(data)):
        raise ValueError("binning values contain non-finite entries")
    edges = np.unique(np.quantile(data, np.linspace(0.0, 1.0, count + 1)))
    if edges.size == 1:
        width = max(abs(float(edges[0])) * 1.0e-12, 1.0e-15)
        edges = np.asarray([edges[0] - width, edges[0] + width])
    return edges


def _numeric_bins(
    values: np.ndarray,
    count: int,
    metric_builder,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = _quantile_edges(data, count)
    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        final = index == edges.size - 2
        mask = (data >= lower) & ((data <= upper) if final else (data < upper))
        rows.append(
            {
                "bin": int(index),
                "lower": float(lower),
                "upper": float(upper),
                "right_inclusive": bool(final),
                **metric_builder(mask),
            }
        )
    return {"edges": edges.tolist(), "bins": rows}


def summarize_limiter_pair_switching(
    base_scales: np.ndarray,
    perturbed_scales: np.ndarray,
    base_controlling: np.ndarray,
    perturbed_controlling: np.ndarray,
    *,
    material_active_scale: float,
    material_scale_change: float,
) -> dict[str, Any]:
    base = np.asarray(base_scales, dtype=np.float64).reshape(-1)
    perturbed = np.asarray(perturbed_scales, dtype=np.float64).reshape(-1)
    base_control = np.asarray(base_controlling, dtype=np.int64).reshape(-1)
    perturbed_control = np.asarray(perturbed_controlling, dtype=np.int64).reshape(-1)
    if not (
        base.shape == perturbed.shape == base_control.shape == perturbed_control.shape
    ):
        raise ValueError("paired limiter arrays must have equal one-dimensional shapes")
    if not 0.0 < material_active_scale < 1.0:
        raise ValueError("material_active_scale must be in (0, 1)")
    if material_scale_change <= 0.0:
        raise ValueError("material_scale_change must be positive")

    exact_base = base < 1.0
    exact_perturbed = perturbed < 1.0
    material_base = base < material_active_scale
    material_perturbed = perturbed < material_active_scale
    scale_change = np.abs(perturbed - base)
    controller_changed = base_control != perturbed_control
    both_exact = exact_base & exact_perturbed
    quantiles = (0.0, 0.5, 0.9, 0.99, 1.0)
    values = np.quantile(scale_change, quantiles)
    return {
        "pair_count": int(base.size),
        "definitions": {
            "exact_active": "limiter scale < 1.0",
            "material_active": f"limiter scale < {material_active_scale:.12g}",
            "material_scale_change": (
                f"absolute paired scale change > {material_scale_change:.12g}"
            ),
        },
        "base_exact_activation_rate": float(np.mean(exact_base)),
        "perturbed_exact_activation_rate": float(np.mean(exact_perturbed)),
        "exact_active_status_switch_count": int(
            np.count_nonzero(exact_base != exact_perturbed)
        ),
        "exact_active_status_switch_rate": float(
            np.mean(exact_base != exact_perturbed)
        ),
        "base_material_activation_rate": float(np.mean(material_base)),
        "perturbed_material_activation_rate": float(np.mean(material_perturbed)),
        "material_active_status_switch_count": int(
            np.count_nonzero(material_base != material_perturbed)
        ),
        "material_active_status_switch_rate": float(
            np.mean(material_base != material_perturbed)
        ),
        "any_exact_scale_change_count": int(np.count_nonzero(scale_change > 0.0)),
        "any_exact_scale_change_rate": float(np.mean(scale_change > 0.0)),
        "material_scale_change_count": int(
            np.count_nonzero(scale_change > material_scale_change)
        ),
        "material_scale_change_rate": float(
            np.mean(scale_change > material_scale_change)
        ),
        "controlling_species_switch_count": int(np.count_nonzero(controller_changed)),
        "controlling_species_switch_rate": float(np.mean(controller_changed)),
        "controlling_species_switch_rate_when_both_exact_active": (
            float(np.mean(controller_changed[both_exact]))
            if np.any(both_exact)
            else float("nan")
        ),
        "absolute_scale_change_quantiles": {
            str(q): float(value) for q, value in zip(quantiles, values)
        },
    }


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    batch = load_fluent_di_jvp_dataset(args.source)
    n_pairs = batch.x.shape[0]
    combined_states = np.concatenate((batch.x, batch.x_perturbed), axis=0)
    combined_dt = np.concatenate((batch.dt, batch.dt), axis=0)
    output, manifest, inference = infer_artifact(
        args.artifact_dir,
        combined_states,
        combined_dt,
        device=args.device,
        batch_size=args.batch_size,
    )
    artifact_names = tuple(str(name) for name in manifest.get("species_names", ()))
    if tuple(map(canonical_species_name, batch.species_names)) != tuple(
        map(canonical_species_name, artifact_names)
    ):
        raise ValueError("artifact and Fluent JVP species orders differ")

    delta_temperature, raw_delta_y = split_artifact_output(
        output, len(batch.species_names)
    )
    base_delta_t = delta_temperature[:n_pairs] if delta_temperature is not None else None
    perturbed_delta_t = (
        delta_temperature[n_pairs:] if delta_temperature is not None else None
    )
    base_raw_delta_y = raw_delta_y[:n_pairs]
    perturbed_raw_delta_y = raw_delta_y[n_pairs:]
    base_delta_y, base_scales, base_controlling = apply_deployment_hard_limiter(
        batch.x[:, 2:], base_raw_delta_y, safety=args.limiter_safety
    )
    perturbed_delta_y, perturbed_scales, perturbed_controlling = (
        apply_deployment_hard_limiter(
            batch.x_perturbed[:, 2:],
            perturbed_raw_delta_y,
            safety=args.limiter_safety,
        )
    )

    base_endpoint = _endpoint(batch.x, base_delta_t, base_delta_y)
    perturbed_endpoint = _endpoint(
        batch.x_perturbed, perturbed_delta_t, perturbed_delta_y
    )
    endpoint_target_jvp = batch.jvp_target
    endpoint_prediction_jvp = _finite_difference(
        perturbed_endpoint, base_endpoint, batch.epsilon
    )

    true_base_rate = (batch.phi_di_x[:, 2:] - batch.x[:, 2:]) / batch.dt[:, None]
    true_perturbed_rate = (
        batch.phi_di_x_perturbed[:, 2:] - batch.x_perturbed[:, 2:]
    ) / batch.dt[:, None]
    predicted_base_rate = base_delta_y / batch.dt[:, None]
    predicted_perturbed_rate = perturbed_delta_y / batch.dt[:, None]
    source_target_jvp = _finite_difference(
        true_perturbed_rate, true_base_rate, batch.epsilon
    )
    source_prediction_jvp = _finite_difference(
        predicted_perturbed_rate, predicted_base_rate, batch.epsilon
    )

    predicts_temperature = delta_temperature is not None
    metrics = _paired_metrics(
        endpoint_target_jvp,
        endpoint_prediction_jvp,
        source_target_jvp,
        source_prediction_jvp,
        batch.species_names,
        predicts_temperature=predicts_temperature,
        sign_zero_tolerance=args.sign_zero_tolerance,
    )
    reactive = (
        np.max(np.abs(batch.phi_di_x[:, 2:] - batch.x[:, 2:]), axis=1)
        > args.reactive_threshold
    )
    direction_type = batch.sample_metadata.get("direction_type")
    if direction_type is None:
        direction_type = np.full(n_pairs, "unknown", dtype=object)
    source_jvp_magnitude = np.linalg.norm(source_target_jvp, axis=1)

    def build(mask: np.ndarray) -> dict[str, Any]:
        return _subset_metrics(
            mask,
            endpoint_target_jvp,
            endpoint_prediction_jvp,
            source_target_jvp,
            source_prediction_jvp,
            batch.species_names,
            predicts_temperature=predicts_temperature,
            sign_zero_tolerance=args.sign_zero_tolerance,
        )

    condition_summaries = {
        "direction_type": _categorical_bins(np.asarray(direction_type), build),
        "temperature": _numeric_bins(batch.x[:, 0], args.condition_bins, build),
        "target_source_jvp_l2_magnitude": _numeric_bins(
            source_jvp_magnitude, args.condition_bins, build
        ),
        "reactive_status": [
            {"value": "nonreactive", **build(~reactive)},
            {"value": "reactive", **build(reactive)},
        ],
    }

    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "source": str(args.source.resolve()),
        "artifact_dir": str(args.artifact_dir.resolve()),
        "provenance": dict(batch.provenance),
        "sampling": {
            "pair_count": int(n_pairs),
            "species_count": int(len(batch.species_names)),
            "direction_types": sorted(set(np.asarray(direction_type).astype(str))),
            "epsilon_min": float(np.min(batch.epsilon)),
            "epsilon_max": float(np.max(batch.epsilon)),
            "dt_min": float(np.min(batch.dt)),
            "dt_max": float(np.max(batch.dt)),
        },
        "artifact_contract": {
            "output": manifest.get("output"),
            "hard_layer": manifest.get("hard_layer"),
            "training_variant": manifest.get("training_config", {}).get("variant"),
            "predicts_temperature": bool(predicts_temperature),
        },
        "jvp_definition": {
            "endpoint": "(Phi(x + epsilon*v) - Phi(x)) / epsilon",
            "source": (
                "(((Phi_Y(x + epsilon*v)-Y_perturbed)/dt) - "
                "((Phi_Y(x)-Y)/dt)) / epsilon"
            ),
            "deployment_stage": "post-artifact, post-UDF-hard-limiter",
        },
        "reactivity": {
            "definition": "max(abs(Phi_DI_Y(x)-Y)) > threshold",
            "threshold": float(args.reactive_threshold),
            "reactive_count": int(np.count_nonzero(reactive)),
            "reactive_fraction": float(np.mean(reactive)),
        },
        "metrics": metrics,
        "condition_summaries": condition_summaries,
        "limiter_pair_switching": summarize_limiter_pair_switching(
            base_scales,
            perturbed_scales,
            base_controlling,
            perturbed_controlling,
            material_active_scale=args.material_limiter_scale,
            material_scale_change=args.material_scale_change,
        ),
        "runtime": {
            "inference": inference,
            "total_evaluator_seconds": float(time.perf_counter() - started),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--condition-bins", type=int, default=4)
    parser.add_argument("--reactive-threshold", type=float, default=1.0e-12)
    parser.add_argument("--sign-zero-tolerance", type=float, default=0.0)
    parser.add_argument("--limiter-safety", type=float, default=DEFAULT_LIMITER_SAFETY)
    parser.add_argument("--material-limiter-scale", type=float, default=0.9999)
    parser.add_argument("--material-scale-change", type=float, default=1.0e-4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.condition_bins <= 0:
        raise ValueError("batch size and condition-bin count must be positive")
    if args.reactive_threshold < 0.0 or args.sign_zero_tolerance < 0.0:
        raise ValueError("thresholds must be non-negative")
    report = _json_compatible(run_evaluation(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "pair_count": report["sampling"]["pair_count"],
                "endpoint_species_jvp": report["metrics"]["endpoint_jvp"][
                    "species"
                ]["overall"],
                "source_jvp": report["metrics"]["delta_y_over_dt_source_jvp"][
                    "overall"
                ],
                "limiter_pair_switching": report["limiter_pair_switching"],
                "runtime": report["runtime"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
