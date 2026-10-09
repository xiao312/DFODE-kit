"""Lineage-preserving, bounded interpolation and perturbation of 1D samples."""
from __future__ import annotations

import numpy as np


def validate_config(config, available_snapshots):
    if config["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema")
    training, validation = config["train_snapshots"], config["validation_snapshots"]
    if not training or not validation or set(training) & set(validation):
        raise ValueError("Training and validation snapshots must be nonempty and disjoint")
    if len(set(training)) != len(training) or len(set(validation)) != len(validation):
        raise ValueError("Duplicate snapshot group")
    if not (set(training) | set(validation)) <= set(available_snapshots):
        raise ValueError("A configured snapshot is absent")
    for name in ("train_count", "validation_count", "wall_seconds"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config["train_count"] > 200000 or config["validation_count"] > 20000 or config["wall_seconds"] > 3600:
        raise ValueError("Plan exceeds bounded benchmark limits")
    for name in ("interval_s", "cvode_rtol", "cvode_atol", "minimum_temperature_K"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("temperature_perturbation_K", "species_exponent_perturbation", "nitrogen_range_padding"):
        if not np.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be nonnegative and finite")
    if config["species_exponent_perturbation"] >= 1:
        raise ValueError("Perturbation exponents must remain positive")
    if "pressure_bounds_Pa" in config:
        bounds = np.asarray(config["pressure_bounds_Pa"], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1]:
            raise ValueError("pressure_bounds_Pa must be two increasing positive finite bounds")


def sample_split(source, species_names, config, split):
    if split not in ("train", "validation"):
        raise ValueError("Expected train or validation split")
    groups = config[f"{split}_snapshots"]
    count = config[f"{split}_count"]
    rng = np.random.default_rng(config["seed"] + (split == "validation"))
    pressure_rng = np.random.default_rng(config["seed"] + 1000 + (split == "validation"))
    gas_states = source["states"]
    rows_by_group = {name: np.flatnonzero(source["snapshot"] == name) for name in groups}
    for name, rows in rows_by_group.items():
        rows_by_group[name] = rows[np.argsort(source["coordinate"][rows], kind="stable")]
        if len(rows) < 2:
            raise ValueError("Each snapshot needs at least two spatial samples")
    source_rows = np.concatenate(list(rows_by_group.values()))
    minimum_temperature = float(gas_states[source_rows, 0].min())
    maximum_temperature = float(gas_states[source_rows, 0].max())
    nitrogen = species_names.index("N2")
    argon = species_names.index("AR") if "AR" in species_names else None
    nitrogen_range = (float(gas_states[source_rows, nitrogen + 2].min()),
                      float(gas_states[source_rows, nitrogen + 2].max()))
    padding = config["nitrogen_range_padding"]
    accepted, lineage, attempts = [], [], 0
    while len(accepted) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("Augmentation acceptance below 1%; review filters instead of looping")
        group = groups[int(rng.integers(len(groups)))]
        rows = rows_by_group[group]
        local = gas_states[rows]
        target_temperature = rng.uniform(minimum_temperature, maximum_temperature)
        # Only interpolate adjacent spatial points in the same snapshot. Multiple
        # crossings are possible in a non-monotone profile; choose one explicitly.
        crossings = np.flatnonzero((local[:-1, 0] - target_temperature) *
                                   (local[1:, 0] - target_temperature) <= 0)
        if not len(crossings):
            continue
        left = int(crossings[int(rng.integers(len(crossings)))])
        difference = local[left + 1, 0] - local[left, 0]
        fraction = float((target_temperature - local[left, 0]) / difference) if difference else .5
        base = (1 - fraction) * local[left] + fraction * local[left + 1]
        state = base.copy()
        state[0] += rng.uniform(-1, 1) * config["temperature_perturbation_K"]
        exponent = 1 + rng.uniform(-1, 1, len(species_names)) * config["species_exponent_perturbation"]
        fractions = base[2:] ** exponent
        if argon is not None:
            inert = float(base[argon + 2])
            fractions[argon] = 0
            fractions *= (1 - inert) / fractions.sum()
            fractions[argon] = inert
        else:
            fractions /= fractions.sum()
        state[2:] = fractions
        if not config["minimum_temperature_K"] <= state[0] <= maximum_temperature + config["temperature_perturbation_K"]:
            continue
        if not nitrogen_range[0] * (1 - padding) <= fractions[nitrogen] <= nitrogen_range[1] * (1 + padding):
            continue
        if "pressure_bounds_Pa" in config:
            state[1] = pressure_rng.uniform(*config["pressure_bounds_Pa"])
        accepted.append(state)
        lineage.append((group, int(rows[left]), int(rows[left + 1]), fraction))
    return np.asarray(accepted), {
        "snapshot": np.array([entry[0] for entry in lineage]),
        "left_source_row": np.array([entry[1] for entry in lineage]),
        "right_source_row": np.array([entry[2] for entry in lineage]),
        "interpolation_fraction": np.array([entry[3] for entry in lineage]),
    }, {"attempts": attempts, "accepted": count, "acceptance_fraction": count / attempts,
        "nitrogen_range": nitrogen_range, "source_temperature_range": [minimum_temperature, maximum_temperature]}
