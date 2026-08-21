from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CONDITION_NAMES = (
    "dt",
    "temperature",
    "mixture_fraction",
    "progress_variable",
)


def _as_float_array(values, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0:
        result = result.reshape(1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    true = _as_float_array(y_true, name="y_true")
    pred = _as_float_array(y_pred, name="y_pred")
    if true.shape != pred.shape:
        raise ValueError(
            f"shape mismatch: y_true has {true.shape}, y_pred has {pred.shape}"
        )
    return true, pred


def normalized_mae(y_true, y_pred, *, zero_tolerance: float = 0.0) -> float:
    """Return sum(abs(error)) / sum(abs(target)).

    A NaN is returned when the target norm is at or below ``zero_tolerance``;
    reporting zero in that case would incorrectly imply an accurate prediction.
    """

    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    true, pred = _validate_pair(y_true, y_pred)
    denominator = float(np.sum(np.abs(true), dtype=np.float64))
    if denominator <= zero_tolerance:
        return float("nan")
    numerator = float(np.sum(np.abs(pred - true), dtype=np.float64))
    return numerator / denominator


def cosine_similarity(y_true, y_pred) -> float:
    true, pred = _validate_pair(y_true, y_pred)
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)
    denominator = float(np.linalg.norm(true_flat) * np.linalg.norm(pred_flat))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(true_flat, pred_flat) / denominator)


def regression_summary(y_true, y_pred) -> dict[str, float | int]:
    true, pred = _validate_pair(y_true, y_pred)
    error = pred - true
    return {
        "sample_count": int(true.shape[0]),
        "entry_count": int(true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "nmae": normalized_mae(true, pred),
        "cosine": cosine_similarity(true, pred),
    }


def density_increment_nmae(
    density_current,
    density_true,
    density_pred,
) -> float:
    current = _as_float_array(density_current, name="density_current")
    true, pred = _validate_pair(density_true, density_pred)
    if current.shape != true.shape:
        raise ValueError("density_current must match target density shape")
    return normalized_mae(true - current, pred - current)


def species_source_from_increment(
    mass_fractions_current,
    mass_fractions_next,
    density,
    dt,
) -> np.ndarray:
    """Convert a mass-fraction endpoint into Fluent species source units.

    The returned source is ``rho * (Y_next - Y_current) / dt``. Density and
    timestep are sample-wise arrays and all arithmetic is float64.
    """

    current, next_state = _validate_pair(
        mass_fractions_current, mass_fractions_next
    )
    if current.ndim != 2:
        raise ValueError("mass fractions must have shape (n_samples, n_species)")
    rho = _as_float_array(density, name="density").reshape(-1)
    timestep = _as_float_array(dt, name="dt").reshape(-1)
    if rho.shape[0] != current.shape[0] or timestep.shape[0] != current.shape[0]:
        raise ValueError("density and dt must have one value per sample")
    if np.any(rho <= 0.0):
        raise ValueError("density must be positive")
    if np.any(timestep <= 0.0):
        raise ValueError("dt must be positive")
    return (rho / timestep)[:, None] * (next_state - current)


def endpoint_from_delta(
    current_state,
    delta_temperature,
    delta_mass_fractions,
) -> np.ndarray:
    """Reconstruct a ``T,p,Y`` endpoint from a delta_T + delta_Y output."""

    current = _as_float_array(current_state, name="current_state")
    delta_y = _as_float_array(
        delta_mass_fractions, name="delta_mass_fractions"
    )
    delta_t = _as_float_array(
        delta_temperature, name="delta_temperature"
    ).reshape(-1)
    if current.ndim != 2 or current.shape[1] < 3:
        raise ValueError("current_state must have shape (n_samples, 2 + n_species)")
    if delta_y.shape != (current.shape[0], current.shape[1] - 2):
        raise ValueError("delta_mass_fractions does not match current_state")
    if delta_t.shape[0] != current.shape[0]:
        raise ValueError("delta_temperature must have one value per sample")
    endpoint = current.copy()
    endpoint[:, 0] += delta_t
    endpoint[:, 2:] += delta_y
    return endpoint


def species_source_summary(
    source_true,
    source_pred,
    species_names: Sequence[str] | None = None,
    *,
    reactive_threshold: float = 0.0,
    reactive_mask=None,
) -> dict:
    true, pred = _validate_pair(source_true, source_pred)
    if true.ndim != 2:
        raise ValueError("species sources must have shape (n_samples, n_species)")
    if reactive_threshold < 0.0:
        raise ValueError("reactive_threshold must be non-negative")

    names = (
        list(species_names)
        if species_names is not None
        else [f"species_{index}" for index in range(true.shape[1])]
    )
    if len(names) != true.shape[1]:
        raise ValueError("species_names does not match source width")

    if reactive_mask is None:
        active = np.max(np.abs(true), axis=1) > reactive_threshold
    else:
        active = np.asarray(reactive_mask, dtype=bool).reshape(-1)
        if active.shape[0] != true.shape[0]:
            raise ValueError("reactive_mask must have one value per sample")

    reactive = (
        regression_summary(true[active], pred[active])
        if np.any(active)
        else _empty_summary()
    )
    return {
        "overall": regression_summary(true, pred),
        "reactive": reactive,
        "reactive_cell_count": int(np.count_nonzero(active)),
        "reactive_cell_fraction": float(np.mean(active)),
        "reactive_threshold": float(reactive_threshold),
        "per_species": [
            {
                "species": name,
                **regression_summary(true[:, index], pred[:, index]),
            }
            for index, name in enumerate(names)
        ],
    }


def binned_regression_summary(
    y_true,
    y_pred,
    condition,
    bin_edges,
) -> list[dict]:
    true, pred = _validate_pair(y_true, y_pred)
    values = _as_float_array(condition, name="condition").reshape(-1)
    edges = _as_float_array(bin_edges, name="bin_edges").reshape(-1)
    if values.shape[0] != true.shape[0]:
        raise ValueError("condition must have one value per sample")
    if edges.shape[0] < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("bin_edges must be strictly increasing")

    rows = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == edges.shape[0] - 2:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        metrics = (
            regression_summary(true[mask], pred[mask])
            if np.any(mask)
            else _empty_summary()
        )
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "right_inclusive": index == edges.shape[0] - 2,
                **metrics,
            }
        )
    return rows


def conditioned_source_summaries(
    source_true,
    source_pred,
    *,
    bin_edges: Mapping[str, Sequence[float]],
    dt=None,
    temperature=None,
    mixture_fraction=None,
    progress_variable=None,
) -> dict[str, list[dict]]:
    conditions = {
        "dt": dt,
        "temperature": temperature,
        "mixture_fraction": mixture_fraction,
        "progress_variable": progress_variable,
    }
    unknown = set(bin_edges) - set(CONDITION_NAMES)
    if unknown:
        raise ValueError(f"unknown condition bin names: {sorted(unknown)}")

    result = {}
    for name, values in conditions.items():
        if values is None:
            continue
        if name not in bin_edges:
            raise ValueError(f"missing bin edges for {name}")
        result[name] = binned_regression_summary(
            source_true, source_pred, values, bin_edges[name]
        )
    return result


def limiter_conditioned_summary(
    source_true,
    source_pred,
    limiter_active,
) -> dict:
    true, pred = _validate_pair(source_true, source_pred)
    active = np.asarray(limiter_active, dtype=bool).reshape(-1)
    if active.shape[0] != true.shape[0]:
        raise ValueError("limiter_active must have one value per sample")
    return {
        "activation_count": int(np.count_nonzero(active)),
        "activation_rate": float(np.mean(active)),
        "active": (
            regression_summary(true[active], pred[active])
            if np.any(active)
            else _empty_summary()
        ),
        "inactive": (
            regression_summary(true[~active], pred[~active])
            if np.any(~active)
            else _empty_summary()
        ),
    }


def prediction_stage_summary(
    target_endpoint,
    raw_network_endpoint,
    post_positivity_endpoint,
    deployed_source_true,
    deployed_source_pred,
    *,
    species_names: Sequence[str] | None = None,
    reactive_threshold: float = 0.0,
    reactive_mask=None,
    limiter_active=None,
) -> dict:
    """Separate endpoint accuracy from the source actually deployed in Fluent.

    ``raw_network_endpoint`` is reconstructed before scalar temperature
    calibration and positivity handling. ``post_positivity_endpoint`` is the
    state after all endpoint corrections. ``deployed_source_pred`` is the
    final source supplied to Fluent and can therefore include density, dt, or
    source-level processing absent from either endpoint.
    """

    target, raw = _validate_pair(target_endpoint, raw_network_endpoint)
    target_again, corrected = _validate_pair(
        target_endpoint, post_positivity_endpoint
    )
    if target.ndim != 2 or target.shape[1] < 3:
        raise ValueError("endpoints must have shape (n_samples, 2 + n_species)")
    if target_again.shape != target.shape:
        raise ValueError("endpoint shapes must match")
    names = (
        list(species_names)
        if species_names is not None
        else [f"species_{index}" for index in range(target.shape[1] - 2)]
    )
    if len(names) != target.shape[1] - 2:
        raise ValueError("species_names does not match endpoint width")

    result = {
        "raw_network_endpoint": _endpoint_metrics(target, raw, names),
        "post_positivity_endpoint": _endpoint_metrics(
            target, corrected, names
        ),
        "deployed_source": species_source_summary(
            deployed_source_true,
            deployed_source_pred,
            names,
            reactive_threshold=reactive_threshold,
            reactive_mask=reactive_mask,
        ),
        "raw_to_post_correction": regression_summary(raw, corrected),
    }
    if limiter_active is not None:
        result["deployed_source_by_limiter"] = limiter_conditioned_summary(
            deployed_source_true, deployed_source_pred, limiter_active
        )
    return result


def summarize_fluent_deployment(
    *,
    density_current,
    density_true,
    density_pred,
    mixture_molecular_weight_true,
    mixture_molecular_weight_pred,
    enthalpy_true,
    enthalpy_pred,
    heat_release_true,
    heat_release_pred,
    species_source_true,
    species_source_pred,
    species_names: Sequence[str] | None = None,
    reactive_threshold: float = 0.0,
    reactive_mask=None,
    condition_bin_edges: Mapping[str, Sequence[float]] | None = None,
    dt=None,
    temperature=None,
    mixture_fraction=None,
    progress_variable=None,
    limiter_active=None,
    target_endpoint=None,
    raw_network_endpoint=None,
    post_positivity_endpoint=None,
) -> dict:
    density_target, density_prediction = _validate_pair(
        density_true, density_pred
    )
    density_now = _as_float_array(
        density_current, name="density_current"
    )
    if density_now.shape != density_target.shape:
        raise ValueError("density_current must match target density shape")

    result = {
        "density_increment": regression_summary(
            density_target - density_now,
            density_prediction - density_now,
        ),
        "mixture_molecular_weight": regression_summary(
            mixture_molecular_weight_true, mixture_molecular_weight_pred
        ),
        "enthalpy": regression_summary(enthalpy_true, enthalpy_pred),
        "heat_release": regression_summary(
            heat_release_true, heat_release_pred
        ),
        "species_source": species_source_summary(
            species_source_true,
            species_source_pred,
            species_names,
            reactive_threshold=reactive_threshold,
            reactive_mask=reactive_mask,
        ),
    }
    result["density_increment"]["nmae"] = density_increment_nmae(
        density_now, density_target, density_prediction
    )

    if condition_bin_edges is not None:
        result["species_source_by_condition"] = conditioned_source_summaries(
            species_source_true,
            species_source_pred,
            bin_edges=condition_bin_edges,
            dt=dt,
            temperature=temperature,
            mixture_fraction=mixture_fraction,
            progress_variable=progress_variable,
        )
    if limiter_active is not None:
        result["species_source_by_limiter"] = limiter_conditioned_summary(
            species_source_true, species_source_pred, limiter_active
        )
    endpoint_values = (
        target_endpoint,
        raw_network_endpoint,
        post_positivity_endpoint,
    )
    if any(value is not None for value in endpoint_values):
        if not all(value is not None for value in endpoint_values):
            raise ValueError(
                "target_endpoint, raw_network_endpoint, and "
                "post_positivity_endpoint must be supplied together"
            )
        result["prediction_stages"] = prediction_stage_summary(
            target_endpoint,
            raw_network_endpoint,
            post_positivity_endpoint,
            species_source_true,
            species_source_pred,
            species_names=species_names,
            reactive_threshold=reactive_threshold,
            reactive_mask=reactive_mask,
            limiter_active=limiter_active,
        )
    return result


def _endpoint_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    species_names: Sequence[str],
) -> dict:
    target_y = target[:, 2:]
    prediction_y = prediction[:, 2:]
    negative = prediction_y < 0.0
    return {
        "overall": regression_summary(target, prediction),
        "temperature": regression_summary(target[:, 0], prediction[:, 0]),
        "pressure": regression_summary(target[:, 1], prediction[:, 1]),
        "species": {
            "overall": regression_summary(target_y, prediction_y),
            "per_species": [
                {
                    "species": name,
                    **regression_summary(
                        target_y[:, index], prediction_y[:, index]
                    ),
                }
                for index, name in enumerate(species_names)
            ],
        },
        "negative_species_entry_rate": float(np.mean(negative)),
        "negative_cell_rate": float(np.mean(np.any(negative, axis=1))),
        "mean_abs_mass_sum_error": float(
            np.mean(np.abs(np.sum(prediction_y, axis=1) - 1.0))
        ),
    }


def _empty_summary() -> dict[str, float | int]:
    return {
        "sample_count": 0,
        "entry_count": 0,
        "mae": float("nan"),
        "rmse": float("nan"),
        "nmae": float("nan"),
        "cosine": float("nan"),
    }
