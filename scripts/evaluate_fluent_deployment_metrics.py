#!/usr/bin/env python3
"""Evaluate a DFODE artifact against Fluent-native direct-integration labels.

The exported model is evaluated exactly as a deployment artifact: float32
thermochemical inputs, optional float64 current-species input, and float64
species increments. The final positivity limiter mirrors the Fluent UDF,
including its 0.999999 safety factor.

Mechanism thermodynamics are optional offline diagnostics. When available,
Cantera is used only to evaluate the mechanism's NASA polynomials; it never
generates labels, and no bitwise equivalence with Fluent thermodynamics is
claimed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch

from dfode_kit.evaluation.fluent_deployment import (
    conditioned_source_summaries,
    prediction_stage_summary,
    regression_summary,
    species_source_from_increment,
    species_source_summary,
    summarize_fluent_deployment,
)


SCHEMA = "dfode.fluent_deployment_evaluation"
FLUENT_BACKENDS = {
    "ansys-fluent-native-di",
    "fluent-native-direct-integration",
    "fluent-native-di",
}
CANONICAL_LABEL_BACKENDS = (
    "fluent-native-di",
    "cantera-cvode-bridge",
)
NON_FLUENT_LABEL_TOKENS = ("cantera", "cvode", "sundials")
DEFAULT_LIMITER_SAFETY = 0.999999


@dataclass(frozen=True)
class EvaluationData:
    current: np.ndarray
    target: np.ndarray
    dt: np.ndarray
    species_names: tuple[str, ...]
    source_index: np.ndarray
    mixture_fraction: np.ndarray | None
    progress_variable: np.ndarray | None
    provenance: dict[str, Any]
    selected_rows: np.ndarray
    total_rows: int


def _decode(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def _attribute_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _known_label_provenance(
    handle: h5py.File,
    expected_label_backend: str,
) -> dict[str, Any]:
    attrs = {key: _attribute_value(value) for key, value in handle.attrs.items()}
    backend = str(attrs.get("label_backend", "")).strip().lower()
    label_source = str(attrs.get("label_source", "")).strip().lower()
    schema = str(attrs.get("schema", "")).strip().lower()
    provenance_text = " ".join(
        (
            backend,
            label_source,
            schema,
            str(attrs.get("label_generator", "")).strip().lower(),
            str(attrs.get("integrator", "")).strip().lower(),
        )
    )
    if expected_label_backend == "fluent-native-di":
        if any(token in provenance_text for token in NON_FLUENT_LABEL_TOKENS):
            raise ValueError(
                "Fluent-native deployment evaluation rejects "
                "Cantera/CVODE/SUNDIALS label provenance"
            )
        accepted = (
            backend in FLUENT_BACKENDS
            or "ansys fluent native direct integration" in label_source
            or schema.startswith("dfode.fluent_native_di")
        )
        normalized_backend = "ansys-fluent-native-di"
    elif expected_label_backend == "cantera-cvode-bridge":
        accepted = backend == "cantera-cvode-bridge"
        normalized_backend = "cantera-cvode-bridge"
    else:
        raise ValueError(
            f"unsupported expected label backend: {expected_label_backend}"
        )
    if not accepted:
        raise ValueError(
            "deployment evaluation label provenance does not match the "
            f"requested backend {expected_label_backend!r}; received "
            f"attributes {attrs!r}"
        )
    return {
        "accepted": True,
        "requested_backend": expected_label_backend,
        "normalized_backend": normalized_backend,
        "source_attributes": attrs,
    }


def _dataset_at(group: h5py.Group, path: str) -> h5py.Dataset:
    value = group[path]
    if not isinstance(value, h5py.Dataset):
        raise ValueError(f"HDF5 object is not a dataset: {path}")
    return value


def _optional_values(
    handle: h5py.File,
    pair_group: h5py.Group,
    explicit_key: str | None,
    candidates: tuple[str, ...],
    selection: slice | np.ndarray,
    row_count: int,
) -> np.ndarray | None:
    search: list[tuple[h5py.Group, str]] = []
    if explicit_key:
        if explicit_key.startswith("pairs/"):
            search.append((handle, explicit_key))
        else:
            search.extend(((pair_group, explicit_key), (handle, explicit_key)))
    else:
        for key in candidates:
            search.extend(((pair_group, key), (handle, key)))

    seen: set[tuple[int, str]] = set()
    for owner, key in search:
        identity = (id(owner), key)
        if identity in seen:
            continue
        seen.add(identity)
        if key not in owner:
            continue
        dataset = _dataset_at(owner, key)
        if dataset.shape[0] != row_count:
            raise ValueError(
                f"optional metadata {dataset.name} has {dataset.shape[0]} rows; "
                f"expected {row_count}"
            )
        return np.asarray(dataset[selection], dtype=np.float64).reshape(-1)
    if explicit_key:
        raise KeyError(f"optional metadata key was requested but not found: {explicit_key}")
    return None


def _selection(row_count: int, max_samples: int, seed: int) -> slice | np.ndarray:
    if max_samples <= 0 or max_samples >= row_count:
        return slice(None)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=max_samples, replace=False))


def load_fluent_native_pairs(
    path: Path,
    *,
    max_samples: int = 0,
    sample_seed: int = 260624,
    mixture_fraction_key: str | None = None,
    progress_variable_key: str | None = None,
    expected_label_backend: str = "fluent-native-di",
) -> EvaluationData:
    with h5py.File(path, "r") as handle:
        provenance = _known_label_provenance(handle, expected_label_backend)
        if "pairs" in handle:
            pair_group = handle["pairs"]
            current_key = "current_states"
            target_key = "target_states"
        else:
            pair_group = handle
            current_key = "state_before"
            target_key = "state_after"

        current_ds = _dataset_at(pair_group, current_key)
        target_ds = _dataset_at(pair_group, target_key)
        dt_ds = _dataset_at(pair_group, "dt")
        if current_ds.shape != target_ds.shape or current_ds.ndim != 2:
            raise ValueError("current and target state datasets must have equal 2D shapes")
        row_count = int(current_ds.shape[0])
        if dt_ds.shape != (row_count,):
            raise ValueError("dt must have one scalar per interval pair")
        selection = _selection(row_count, max_samples, sample_seed)
        current = np.asarray(current_ds[selection], dtype=np.float64)
        target = np.asarray(target_ds[selection], dtype=np.float64)
        dt = np.asarray(dt_ds[selection], dtype=np.float64)
        if np.any(~np.isfinite(current)) or np.any(~np.isfinite(target)):
            raise ValueError("state arrays contain non-finite values")
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
            raise ValueError("all Fluent chemistry intervals must be finite and positive")

        species_owner = handle if "species_names" in handle else pair_group
        species_names = _decode(np.asarray(species_owner["species_names"]))
        if current.shape[1] != len(species_names) + 2:
            raise ValueError("state width does not match T, P, and species names")
        if "source_index" in pair_group:
            source_index = np.asarray(
                pair_group["source_index"][selection], dtype=np.int64
            )
        else:
            selected = np.arange(row_count, dtype=np.int64)[selection]
            source_index = np.asarray(selected, dtype=np.int64)

        mixture_fraction = _optional_values(
            handle,
            pair_group,
            mixture_fraction_key,
            ("mixture_fraction", "mixture-fraction", "Z", "z"),
            selection,
            row_count,
        )
        progress_variable = _optional_values(
            handle,
            pair_group,
            progress_variable_key,
            ("progress_variable", "progress-variable", "progress", "c"),
            selection,
            row_count,
        )

    selected_rows = np.arange(row_count, dtype=np.int64)[selection]
    return EvaluationData(
        current=current,
        target=target,
        dt=dt,
        species_names=species_names,
        source_index=source_index,
        mixture_fraction=mixture_fraction,
        progress_variable=progress_variable,
        provenance=provenance,
        selected_rows=np.asarray(selected_rows, dtype=np.int64),
        total_rows=row_count,
    )


def _normalized_species_name(name: str) -> str:
    return name.replace("<S>", "(S)")


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
    current: np.ndarray,
    dt: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    model = torch.jit.load(str(artifact_dir / "model.pt"), map_location=device).eval()
    separate_species_input = (
        manifest.get("hard_layer", {}).get("current_species_source")
        == "separate_input"
    )
    use_cuda_api = str(device).startswith("cuda") and torch.cuda.is_available()

    warm_end = min(batch_size, current.shape[0])
    warm_input = np.concatenate(
        [current[:warm_end], dt[:warm_end, None]], axis=1
    ).astype(np.float32, copy=False)
    with torch.inference_mode():
        _model_call(
            model,
            torch.from_numpy(warm_input).to(device),
            torch.from_numpy(
                np.ascontiguousarray(current[:warm_end, 2:], dtype=np.float64)
            ).to(device),
            separate_species_input,
        )
    if use_cuda_api:
        torch.cuda.synchronize()

    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, current.shape[0], batch_size):
            end = min(begin + batch_size, current.shape[0])
            model_input = np.concatenate(
                [current[begin:end], dt[begin:end, None]], axis=1
            ).astype(np.float32, copy=False)
            result = _model_call(
                model,
                torch.from_numpy(model_input).to(device),
                torch.from_numpy(
                    np.ascontiguousarray(current[begin:end, 2:], dtype=np.float64)
                ).to(device),
                separate_species_input,
            )
            outputs.append(result.detach().cpu().numpy().astype(np.float64, copy=False))
    if use_cuda_api:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = np.concatenate(outputs, axis=0)
    runtime = {
        "device": str(device),
        "batch_size": int(batch_size),
        "sample_count": int(current.shape[0]),
        "elapsed_seconds": float(elapsed),
        "samples_per_second": float(current.shape[0] / max(elapsed, 1.0e-300)),
        "microseconds_per_sample": float(1.0e6 * elapsed / current.shape[0]),
        "warmup_excluded": True,
    }
    return output, manifest, runtime


def split_artifact_output(
    output: np.ndarray,
    n_species: int,
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
    """Mirror the Fluent UDF global limiter in float64.

    Returns limited increments, row scales, and controlling species indices.
    A controlling index of -1 means the limiter did not activate.
    """

    current = np.asarray(current_species, dtype=np.float64)
    delta = np.asarray(delta_y, dtype=np.float64)
    if current.shape != delta.shape or current.ndim != 2:
        raise ValueError("current_species and delta_y must have equal 2D shapes")
    if not 0.0 < safety <= 1.0:
        raise ValueError("limiter safety must be in (0, 1]")
    candidates = np.full(delta.shape, np.inf, dtype=np.float64)
    consuming = delta < 0.0
    candidates[consuming] = (
        safety * current[consuming] / (-delta[consuming])
    )
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
    result = current.copy()
    if delta_temperature is not None:
        result[:, 0] += delta_temperature
    result[:, 2:] += delta_y
    return result


def _stage_endpoint_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    true_rate: np.ndarray,
    predicted_rate: np.ndarray,
    species_names: tuple[str, ...],
    reactive_mask: np.ndarray,
) -> dict[str, Any]:
    stage = prediction_stage_summary(
        target,
        prediction,
        prediction,
        true_rate,
        predicted_rate,
        species_names=species_names,
        reactive_mask=reactive_mask,
    )
    return stage["raw_network_endpoint"]


def _parse_edges(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    edges = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin edges must be finite and strictly increasing")
    return edges


def _automatic_edges(values: np.ndarray, bins: int, *, logarithmic: bool) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    if minimum == maximum:
        width = max(abs(minimum) * 1.0e-6, 1.0e-15)
        return np.asarray([minimum - width, maximum + width], dtype=np.float64)
    if logarithmic and minimum > 0.0:
        return np.geomspace(minimum, maximum, bins + 1)
    return np.linspace(minimum, maximum, bins + 1)


def _limiter_summary(
    scales: np.ndarray,
    controlling: np.ndarray,
    species_names: tuple[str, ...],
    raw_delta: np.ndarray,
    limited_delta: np.ndarray,
) -> dict[str, Any]:
    active = scales < 1.0
    quantiles = (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0)
    all_values = np.quantile(scales, quantiles)
    active_values = np.quantile(scales[active], quantiles) if np.any(active) else None
    counts = np.bincount(
        controlling[active], minlength=len(species_names)
    ) if np.any(active) else np.zeros(len(species_names), dtype=np.int64)
    order = np.argsort(counts)[::-1]
    top = [
        {
            "species": species_names[index],
            "count": int(counts[index]),
            "fraction_of_limited": float(counts[index] / np.count_nonzero(active)),
        }
        for index in order
        if counts[index] > 0
    ]
    denominator = float(np.sum(np.abs(raw_delta), dtype=np.float64))
    correction = float(np.sum(np.abs(limited_delta - raw_delta), dtype=np.float64))
    return {
        "implementation": "global-consumption-ratio-minimum",
        "safety_factor": DEFAULT_LIMITER_SAFETY,
        "activation_count": int(np.count_nonzero(active)),
        "activation_rate": float(np.mean(active)),
        "scale_quantiles": {
            str(q): float(value) for q, value in zip(quantiles, all_values)
        },
        "active_scale_quantiles": (
            {str(q): float(value) for q, value in zip(quantiles, active_values)}
            if active_values is not None
            else None
        ),
        "mean_scale": float(np.mean(scales)),
        "minimum_scale": float(np.min(scales)),
        "relative_l1_correction": correction / max(denominator, 1.0e-300),
        "controlling_species": top,
    }


def _mixture_molecular_weight(y: np.ndarray, molecular_weights: np.ndarray) -> np.ndarray:
    mass_sum = np.sum(y, axis=1, dtype=np.float64)
    inverse = np.sum(y / molecular_weights[None, :], axis=1, dtype=np.float64)
    if np.any(~np.isfinite(inverse)) or np.any(inverse <= 0.0):
        raise ValueError("offline thermo received a nonpositive mixture molar density")
    return mass_sum / inverse


def _offline_thermo_diagnostics(
    artifact_dir: Path,
    manifest: dict[str, Any],
    current: np.ndarray,
    target: np.ndarray,
    deployed: np.ndarray,
    dt: np.ndarray,
    species_names: tuple[str, ...],
    *,
    chunk_size: int,
) -> dict[str, Any]:
    try:
        import cantera as ct
    except ImportError as error:
        raise RuntimeError("Cantera is unavailable for optional NASA diagnostics") from error

    mechanism_value = manifest.get("mechanism")
    if not mechanism_value:
        raise RuntimeError("artifact manifest has no mechanism path")
    mechanism = artifact_dir / str(mechanism_value)
    if not mechanism.is_file():
        raise RuntimeError(f"artifact mechanism is missing: {mechanism}")
    gas = ct.Solution(str(mechanism))
    mechanism_names = tuple(gas.species_names)
    normalized = tuple(_normalized_species_name(name) for name in species_names)
    if mechanism_names != normalized:
        raise RuntimeError("artifact mechanism and evaluation species orders differ")

    molecular_weights = np.asarray(gas.molecular_weights, dtype=np.float64)
    gas_constant = float(ct.gas_constant)
    current_y = current[:, 2:]
    target_y = target[:, 2:]
    deployed_y = deployed[:, 2:]
    current_w = _mixture_molecular_weight(current_y, molecular_weights)
    target_w = _mixture_molecular_weight(target_y, molecular_weights)
    deployed_w = _mixture_molecular_weight(deployed_y, molecular_weights)
    current_density = current[:, 1] * current_w / (gas_constant * current[:, 0])
    target_density_labeled_state = (
        target[:, 1] * target_w / (gas_constant * target[:, 0])
    )
    target_density_fixed_pressure = (
        current[:, 1] * target_w / (gas_constant * target[:, 0])
    )
    deployed_density_fixed_pressure = (
        current[:, 1] * deployed_w / (gas_constant * deployed[:, 0])
    )

    current_h = np.empty(current.shape[0], dtype=np.float64)
    target_h = np.empty(current.shape[0], dtype=np.float64)
    deployed_h = np.empty(current.shape[0], dtype=np.float64)
    true_heat_release = np.empty(current.shape[0], dtype=np.float64)
    deployed_heat_release = np.empty(current.shape[0], dtype=np.float64)
    for begin in range(0, current.shape[0], chunk_size):
        end = min(begin + chunk_size, current.shape[0])
        state_count = end - begin

        current_states = ct.SolutionArray(gas, shape=(state_count,))
        current_states.TPY = (
            current[begin:end, 0],
            current[begin:end, 1],
            current_y[begin:end],
        )
        current_species_h = np.asarray(
            current_states.partial_molar_enthalpies, dtype=np.float64
        ) / molecular_weights[None, :]
        current_h[begin:end] = np.sum(
            current_y[begin:end] * current_species_h, axis=1, dtype=np.float64
        )
        true_heat_release[begin:end] = -current_density[begin:end] * np.sum(
            current_species_h * (target_y[begin:end] - current_y[begin:end]),
            axis=1,
            dtype=np.float64,
        ) / dt[begin:end]
        deployed_heat_release[begin:end] = -current_density[begin:end] * np.sum(
            current_species_h * (deployed_y[begin:end] - current_y[begin:end]),
            axis=1,
            dtype=np.float64,
        ) / dt[begin:end]

        target_states = ct.SolutionArray(gas, shape=(state_count,))
        target_states.TPY = (
            target[begin:end, 0], target[begin:end, 1], target_y[begin:end]
        )
        target_species_h = np.asarray(
            target_states.partial_molar_enthalpies, dtype=np.float64
        ) / molecular_weights[None, :]
        target_h[begin:end] = np.sum(
            target_y[begin:end] * target_species_h, axis=1, dtype=np.float64
        )

        deployed_states = ct.SolutionArray(gas, shape=(state_count,))
        deployed_states.TPY = (
            deployed[begin:end, 0],
            deployed[begin:end, 1],
            deployed_y[begin:end],
        )
        deployed_species_h = np.asarray(
            deployed_states.partial_molar_enthalpies, dtype=np.float64
        ) / molecular_weights[None, :]
        deployed_h[begin:end] = np.sum(
            deployed_y[begin:end] * deployed_species_h, axis=1, dtype=np.float64
        )

    return {
        "available": True,
        "model": "artifact-mechanism ideal-gas NASA polynomials via Cantera",
        "cantera_version": str(ct.__version__),
        "mechanism": str(mechanism.resolve()),
        "purpose": "offline deployment diagnostic only",
        "used_to_generate_labels": False,
        "exact_fluent_thermo_equivalence": False,
        "warning": (
            "These NASA/ideal-gas diagnostics are not asserted to be bitwise "
            "or thermodynamically identical to Fluent material properties."
        ),
        "arrays": {
            "current_mixture_molecular_weight": current_w,
            "target_mixture_molecular_weight": target_w,
            "deployed_mixture_molecular_weight": deployed_w,
            "current_density": current_density,
            "target_density_labeled_state": target_density_labeled_state,
            "target_density_fixed_pressure": target_density_fixed_pressure,
            "deployed_density_fixed_pressure": deployed_density_fixed_pressure,
            "target_enthalpy": target_h,
            "deployed_enthalpy": deployed_h,
            "true_heat_release": true_heat_release,
            "deployed_heat_release": deployed_heat_release,
        },
    }


def _density_contract_metrics(
    *,
    density_current: np.ndarray,
    density_target_labeled_state: np.ndarray,
    density_target_fixed_pressure: np.ndarray,
    density_prediction_fixed_pressure: np.ndarray,
    pressure_current: np.ndarray,
    pressure_target: np.ndarray,
) -> dict[str, Any]:
    """Separate labeled-flow density changes from chemistry-only changes.

    The artifact predicts ``delta_T`` and ``delta_Y`` but no pressure update.
    Consequently, comparison against a labeled endpoint evaluated at target
    pressure includes a CFD pressure contribution that the artifact cannot
    control. The fixed-pressure contract evaluates both target and prediction
    at the current input pressure and is the deployment-consistent chemistry
    metric.
    """

    current = np.asarray(density_current, dtype=np.float64)
    target_labeled = np.asarray(
        density_target_labeled_state, dtype=np.float64
    )
    target_fixed = np.asarray(density_target_fixed_pressure, dtype=np.float64)
    prediction_fixed = np.asarray(
        density_prediction_fixed_pressure, dtype=np.float64
    )
    current_pressure = np.asarray(pressure_current, dtype=np.float64)
    target_pressure = np.asarray(pressure_target, dtype=np.float64)
    arrays = (target_labeled, target_fixed, prediction_fixed)
    if any(values.shape != current.shape for values in arrays):
        raise ValueError("all density arrays must have equal shapes")
    if current_pressure.shape != target_pressure.shape or (
        current_pressure.shape != current.shape
    ):
        raise ValueError("pressure arrays must match density arrays")

    pressure_change = target_pressure - current_pressure
    full_labeled_state = {
        "controllable_by_chemistry_artifact": False,
        "target_pressure_contract": "Fluent-native labeled endpoint pressure",
        "prediction_pressure_contract": "current input pressure",
        "warning": (
            "This comparison includes labeled CFD pressure evolution, but the "
            "chemistry artifact predicts only delta_T and delta_Y. It is not a "
            "controllable chemistry-artifact metric."
        ),
        "labeled_pressure_change_pa": {
            "mean": float(np.mean(pressure_change)),
            "mean_absolute": float(np.mean(np.abs(pressure_change))),
            "minimum": float(np.min(pressure_change)),
            "maximum": float(np.max(pressure_change)),
        },
        "endpoint_density": regression_summary(target_labeled, prediction_fixed),
        "density_increment": regression_summary(
            target_labeled - current, prediction_fixed - current
        ),
    }
    fixed_pressure_chemistry = {
        "controllable_by_chemistry_artifact": True,
        "target_pressure_contract": "current input pressure",
        "prediction_pressure_contract": "current input pressure",
        "description": (
            "Deployment-consistent chemistry density comparison with current "
            "pressure held fixed for both Fluent-native target T/Y and predicted T/Y."
        ),
        "endpoint_density": regression_summary(target_fixed, prediction_fixed),
        "density_increment": regression_summary(
            target_fixed - current, prediction_fixed - current
        ),
    }
    return {
        "full_labeled_state_density": full_labeled_state,
        "deployment_consistent_fixed_pressure_chemistry_density": (
            fixed_pressure_chemistry
        ),
    }


def _deprecated_density_increment_alias(
    density_contracts: dict[str, Any],
) -> dict[str, Any]:
    alias_target = (
        "density_contracts.full_labeled_state_density.density_increment"
    )
    return {
        **density_contracts["full_labeled_state_density"]["density_increment"],
        "deprecated": True,
        "alias_of": alias_target,
        "deprecation_reason": (
            "Historical behavior includes target-pressure evolution that the "
            "delta_T/delta_Y chemistry artifact cannot control. Use "
            "density_contracts.deployment_consistent_fixed_pressure_chemistry_density "
            "for deployment assessment."
        ),
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
    data = load_fluent_native_pairs(
        args.source,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
        mixture_fraction_key=args.mixture_fraction_key,
        progress_variable_key=args.progress_variable_key,
        expected_label_backend=args.expected_label_backend,
    )
    model_output, manifest, inference_runtime = infer_artifact(
        args.artifact_dir,
        data.current,
        data.dt,
        device=args.device,
        batch_size=args.batch_size,
    )
    artifact_names = tuple(manifest.get("species_names", ()))
    if tuple(map(_normalized_species_name, data.species_names)) != tuple(
        map(_normalized_species_name, artifact_names)
    ):
        raise ValueError("artifact and Fluent label species orders differ")

    predicted_delta_temperature, raw_delta_y = split_artifact_output(
        model_output, len(data.species_names)
    )
    post_model_delta_y = raw_delta_y.copy()
    deployed_delta_y, limiter_scales, controlling_species = (
        apply_deployment_hard_limiter(
            data.current[:, 2:], raw_delta_y, safety=args.limiter_safety
        )
    )
    raw_endpoint = _endpoint(
        data.current, predicted_delta_temperature, raw_delta_y
    )
    post_model_endpoint = _endpoint(
        data.current, predicted_delta_temperature, post_model_delta_y
    )
    deployed_endpoint = _endpoint(
        data.current, predicted_delta_temperature, deployed_delta_y
    )
    true_delta_y = data.target[:, 2:] - data.current[:, 2:]
    true_rate = true_delta_y / data.dt[:, None]
    raw_rate = raw_delta_y / data.dt[:, None]
    post_model_rate = post_model_delta_y / data.dt[:, None]
    deployed_rate = deployed_delta_y / data.dt[:, None]
    reactive_mask = np.max(np.abs(true_delta_y), axis=1) > args.reactive_threshold
    limiter_active = limiter_scales < 1.0

    temperature_edges = _parse_edges(args.temperature_bin_edges)
    if temperature_edges is None:
        temperature_edges = _automatic_edges(
            data.current[:, 0], args.condition_bins, logarithmic=False
        )
    dt_edges = _parse_edges(args.dt_bin_edges)
    if dt_edges is None:
        dt_edges = _automatic_edges(data.dt, args.condition_bins, logarithmic=True)
    condition_edges: dict[str, np.ndarray] = {
        "temperature": temperature_edges,
        "dt": dt_edges,
    }
    if data.mixture_fraction is not None:
        condition_edges["mixture_fraction"] = _automatic_edges(
            data.mixture_fraction, args.condition_bins, logarithmic=False
        )
    if data.progress_variable is not None:
        condition_edges["progress_variable"] = _automatic_edges(
            data.progress_variable, args.condition_bins, logarithmic=False
        )

    stage_summaries = {
        "raw_model_endpoint": _stage_endpoint_summary(
            data.target,
            raw_endpoint,
            true_rate,
            raw_rate,
            data.species_names,
            reactive_mask,
        ),
        "post_model_positivity_endpoint": _stage_endpoint_summary(
            data.target,
            post_model_endpoint,
            true_rate,
            post_model_rate,
            data.species_names,
            reactive_mask,
        ),
        "deployment_hard_limiter_endpoint": _stage_endpoint_summary(
            data.target,
            deployed_endpoint,
            true_rate,
            deployed_rate,
            data.species_names,
            reactive_mask,
        ),
    }
    rate_summaries = {
        "units": "mass-fraction/s",
        "raw_model": species_source_summary(
            true_rate,
            raw_rate,
            data.species_names,
            reactive_mask=reactive_mask,
        ),
        "post_model_positivity": species_source_summary(
            true_rate,
            post_model_rate,
            data.species_names,
            reactive_mask=reactive_mask,
        ),
        "deployment_hard_limiter": species_source_summary(
            true_rate,
            deployed_rate,
            data.species_names,
            reactive_mask=reactive_mask,
        ),
        "deployment_by_condition": conditioned_source_summaries(
            true_rate,
            deployed_rate,
            bin_edges=condition_edges,
            dt=data.dt,
            temperature=data.current[:, 0],
            mixture_fraction=data.mixture_fraction,
            progress_variable=data.progress_variable,
        ),
    }

    thermo_public: dict[str, Any]
    deployment_metrics: dict[str, Any] | None = None
    if args.thermo == "none":
        thermo_public = {
            "available": False,
            "reason": "disabled by --thermo none",
            "used_to_generate_labels": False,
            "exact_fluent_thermo_equivalence": False,
        }
    else:
        try:
            thermo = _offline_thermo_diagnostics(
                args.artifact_dir,
                manifest,
                data.current,
                data.target,
                deployed_endpoint,
                data.dt,
                data.species_names,
                chunk_size=args.thermo_chunk_size,
            )
            arrays = thermo.pop("arrays")
            true_mass_source = species_source_from_increment(
                data.current[:, 2:],
                data.target[:, 2:],
                arrays["current_density"],
                data.dt,
            )
            deployed_mass_source = species_source_from_increment(
                data.current[:, 2:],
                deployed_endpoint[:, 2:],
                arrays["current_density"],
                data.dt,
            )
            deployment_metrics = summarize_fluent_deployment(
                density_current=arrays["current_density"],
                density_true=arrays["target_density_labeled_state"],
                density_pred=arrays["deployed_density_fixed_pressure"],
                mixture_molecular_weight_true=arrays[
                    "target_mixture_molecular_weight"
                ],
                mixture_molecular_weight_pred=arrays[
                    "deployed_mixture_molecular_weight"
                ],
                enthalpy_true=arrays["target_enthalpy"],
                enthalpy_pred=arrays["deployed_enthalpy"],
                heat_release_true=arrays["true_heat_release"],
                heat_release_pred=arrays["deployed_heat_release"],
                species_source_true=true_mass_source,
                species_source_pred=deployed_mass_source,
                species_names=data.species_names,
                reactive_mask=reactive_mask,
                condition_bin_edges=condition_edges,
                dt=data.dt,
                temperature=data.current[:, 0],
                mixture_fraction=data.mixture_fraction,
                progress_variable=data.progress_variable,
                limiter_active=limiter_active,
                target_endpoint=data.target,
                raw_network_endpoint=raw_endpoint,
                post_positivity_endpoint=post_model_endpoint,
            )
            density_contracts = _density_contract_metrics(
                density_current=arrays["current_density"],
                density_target_labeled_state=arrays[
                    "target_density_labeled_state"
                ],
                density_target_fixed_pressure=arrays[
                    "target_density_fixed_pressure"
                ],
                density_prediction_fixed_pressure=arrays[
                    "deployed_density_fixed_pressure"
                ],
                pressure_current=data.current[:, 1],
                pressure_target=data.target[:, 1],
            )
            deployment_metrics["density_contracts"] = density_contracts
            deployment_metrics["density_increment"] = (
                _deprecated_density_increment_alias(density_contracts)
            )
            thermo_public = thermo
            thermo_public["metrics"] = {
                "mixture_molecular_weight": deployment_metrics[
                    "mixture_molecular_weight"
                ],
                "density_contracts": density_contracts,
                "ideal_gas_density_increment": {
                    **deployment_metrics["density_increment"],
                    "alias_of": (
                        "offline_thermodynamics.metrics.density_contracts."
                        "full_labeled_state_density.density_increment"
                    ),
                },
                "mixture_enthalpy": deployment_metrics["enthalpy"],
                "heat_release": deployment_metrics["heat_release"],
                "fluent_mass_source": deployment_metrics["species_source"],
            }
        except Exception as error:
            if args.thermo == "required":
                raise
            thermo_public = {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
                "used_to_generate_labels": False,
                "exact_fluent_thermo_equivalence": False,
            }

    report = {
        "schema": SCHEMA,
        "schema_version": 2,
        "artifact_dir": str(args.artifact_dir.resolve()),
        "source": str(args.source.resolve()),
        "provenance": data.provenance,
        "sampling": {
            "source_rows": int(data.total_rows),
            "evaluated_rows": int(data.current.shape[0]),
            "max_samples": int(args.max_samples),
            "sample_seed": int(args.sample_seed),
            "selected_row_min": int(np.min(data.selected_rows)),
            "selected_row_max": int(np.max(data.selected_rows)),
        },
        "artifact_contract": {
            "output": manifest.get("output"),
            "hard_layer": manifest.get("hard_layer"),
            "training_variant": manifest.get("training_config", {}).get("variant"),
            "predicts_temperature": predicted_delta_temperature is not None,
            "predicts_pressure": False,
        },
        "stage_semantics": {
            "raw_model_endpoint": (
                "direct TorchScript artifact output before deployment corrections"
            ),
            "post_model_positivity_endpoint": (
                "identical to artifact output because current v17 exports the "
                "model-internal sparse Patankar-positive delta; the unconstrained "
                "pre-Patankar network output is not exposed by the artifact"
            ),
            "deployment_hard_limiter_endpoint": (
                "artifact delta_Y multiplied by the Fluent UDF global limiter"
            ),
            "raw_equals_post_model_for_exported_artifact": True,
        },
        "runtime": {
            "inference": inference_runtime,
            "total_evaluator_seconds": float(time.perf_counter() - started),
        },
        "reactivity": {
            "delta_y_threshold": float(args.reactive_threshold),
            "reactive_count": int(np.count_nonzero(reactive_mask)),
            "reactive_fraction": float(np.mean(reactive_mask)),
        },
        "endpoint_metrics": stage_summaries,
        "delta_y_over_dt": rate_summaries,
        "limiter": _limiter_summary(
            limiter_scales,
            controlling_species,
            data.species_names,
            raw_delta_y,
            deployed_delta_y,
        ),
        "condition_bins": {
            key: values.tolist() for key, values in condition_edges.items()
        },
        "optional_metadata": {
            "mixture_fraction_available": data.mixture_fraction is not None,
            "progress_variable_available": data.progress_variable is not None,
        },
        "offline_thermodynamics": thermo_public,
    }
    if deployment_metrics is not None:
        report["fluent_deployment_metrics"] = deployment_metrics
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=260624)
    parser.add_argument("--reactive-threshold", type=float, default=1.0e-12)
    parser.add_argument("--limiter-safety", type=float, default=DEFAULT_LIMITER_SAFETY)
    parser.add_argument("--condition-bins", type=int, default=6)
    parser.add_argument("--temperature-bin-edges")
    parser.add_argument("--dt-bin-edges")
    parser.add_argument("--mixture-fraction-key")
    parser.add_argument("--progress-variable-key")
    parser.add_argument(
        "--expected-label-backend",
        choices=CANONICAL_LABEL_BACKENDS,
        default="fluent-native-di",
    )
    parser.add_argument("--thermo", choices=("auto", "none", "required"), default="auto")
    parser.add_argument("--thermo-chunk-size", type=int, default=4096)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.thermo_chunk_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.max_samples < 0 or args.condition_bins <= 0:
        raise ValueError("max samples must be non-negative and condition bins positive")
    report = _json_compatible(run_evaluation(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    concise = {
        "output": str(args.output.resolve()),
        "evaluated_rows": report["sampling"]["evaluated_rows"],
        "inference": report["runtime"]["inference"],
        "species_rate": report["delta_y_over_dt"]["deployment_hard_limiter"][
            "overall"
        ],
        "limiter": report["limiter"],
        "offline_thermodynamics": report["offline_thermodynamics"],
    }
    if concise["offline_thermodynamics"].get("metrics"):
        concise["offline_thermodynamics"] = {
            key: value
            for key, value in concise["offline_thermodynamics"].items()
            if key != "metrics"
        } | {"metrics": report["offline_thermodynamics"]["metrics"]}
    print(json.dumps(concise, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
