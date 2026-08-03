from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = "cfd-cell-snapshot-v1"


def _decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _interpolation_temperatures(
    first: float, second: float, spacing: float
) -> np.ndarray:
    lower = min(first, second)
    upper = max(first, second)
    first_index = int(np.floor(lower / spacing)) + 1
    last_index = int(np.ceil(upper / spacing)) - 1
    if last_index < first_index:
        return np.empty(0, dtype=np.float64)
    values = spacing * np.arange(
        first_index, last_index + 1, dtype=np.float64
    )
    return values[(values > lower) & (values < upper)]


def interpolate_temperature_grid_snapshot(
    source_path: str | Path,
    output_path: str | Path,
    *,
    temperature_spacing: float,
    include_original: bool = True,
) -> dict:
    if temperature_spacing <= 0.0:
        raise ValueError("temperature_spacing must be positive")
    source_path = Path(source_path)
    output_path = Path(output_path)
    with h5py.File(source_path, "r") as source:
        if str(source.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError(f"expected {SCHEMA_VERSION}")
        cells = source["cells"]
        states = np.asarray(cells["states"], dtype=np.float64)
        coordinates = np.asarray(cells["coordinates"], dtype=np.float64)
        edges = np.asarray(cells["neighbor_edges"], dtype=np.int64)
        source_cell_index = np.asarray(
            cells["source_cell_index"], dtype=np.int64
        )
        species_names = _decode(source["species_names"][:])
        attrs = dict(source.attrs)

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("neighbor_edges must have shape (n_edges, 2)")
    if np.any(edges < 0) or np.any(edges >= states.shape[0]):
        raise ValueError("neighbor edge index is outside the snapshot")

    state_parts: list[np.ndarray] = []
    coordinate_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []
    fraction_parts: list[np.ndarray] = []
    temperature_parts: list[np.ndarray] = []
    generated_per_edge = np.zeros(edges.shape[0], dtype=np.int32)

    for edge_index, (first_index, second_index) in enumerate(edges):
        first = states[first_index]
        second = states[second_index]
        temperatures = _interpolation_temperatures(
            float(first[0]),
            float(second[0]),
            temperature_spacing,
        )
        if temperatures.size == 0:
            continue
        fractions = (temperatures - first[0]) / (
            second[0] - first[0]
        )
        state_parts.append(
            first[None, :]
            + fractions[:, None] * (second - first)[None, :]
        )
        coordinate_parts.append(
            coordinates[first_index][None, :]
            + fractions[:, None]
            * (
                coordinates[second_index]
                - coordinates[first_index]
            )[None, :]
        )
        parents = np.asarray(
            [
                source_cell_index[first_index],
                source_cell_index[second_index],
            ],
            dtype=np.int64,
        )
        parent_parts.append(
            np.repeat(parents[None, :], temperatures.size, axis=0)
        )
        fraction_parts.append(fractions)
        temperature_parts.append(temperatures)
        generated_per_edge[edge_index] = temperatures.size

    width = states.shape[1]
    generated_states = (
        np.concatenate(state_parts)
        if state_parts
        else np.empty((0, width), dtype=np.float64)
    )
    generated_coordinates = (
        np.concatenate(coordinate_parts)
        if coordinate_parts
        else np.empty((0, coordinates.shape[1]), dtype=np.float64)
    )
    generated_parents = (
        np.concatenate(parent_parts)
        if parent_parts
        else np.empty((0, 2), dtype=np.int64)
    )
    generated_fractions = (
        np.concatenate(fraction_parts)
        if fraction_parts
        else np.empty(0, dtype=np.float64)
    )
    generated_temperatures = (
        np.concatenate(temperature_parts)
        if temperature_parts
        else np.empty(0, dtype=np.float64)
    )

    if include_original:
        output_states = np.concatenate([states, generated_states])
        output_coordinates = np.concatenate(
            [coordinates, generated_coordinates]
        )
        original_parents = np.column_stack(
            [source_cell_index, source_cell_index]
        )
        output_parents = np.concatenate(
            [original_parents, generated_parents]
        )
        output_fractions = np.concatenate(
            [
                np.zeros(states.shape[0], dtype=np.float64),
                generated_fractions,
            ]
        )
        output_source_index = np.concatenate(
            [
                source_cell_index,
                np.full(generated_states.shape[0], -1, dtype=np.int64),
            ]
        )
        original_mask = np.concatenate(
            [
                np.ones(states.shape[0], dtype=np.uint8),
                np.zeros(generated_states.shape[0], dtype=np.uint8),
            ]
        )
    else:
        output_states = generated_states
        output_coordinates = generated_coordinates
        output_parents = generated_parents
        output_fractions = generated_fractions
        output_source_index = np.full(
            generated_states.shape[0], -1, dtype=np.int64
        )
        original_mask = np.zeros(
            generated_states.shape[0], dtype=np.uint8
        )

    mass_sums = output_states[:, 2:].sum(axis=1)
    summary = {
        "source": str(source_path),
        "output": str(output_path),
        "temperature_spacing_k": temperature_spacing,
        "source_states": int(states.shape[0]),
        "neighbor_edges": int(edges.shape[0]),
        "active_edges": int(np.count_nonzero(generated_per_edge)),
        "generated_states": int(generated_states.shape[0]),
        "output_states": int(output_states.shape[0]),
        "include_original": include_original,
        "temperature_min_k": float(output_states[:, 0].min()),
        "temperature_max_k": float(output_states[:, 0].max()),
        "minimum_mass_fraction": float(output_states[:, 2:].min()),
        "maximum_mass_sum_error": float(
            np.max(np.abs(mass_sums - 1.0))
        ),
        "edge_generated_count_percentiles": {
            str(value): float(
                np.percentile(generated_per_edge, value)
            )
            for value in (50, 75, 90, 95, 99, 100)
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as output:
        for name, value in attrs.items():
            output.attrs[name] = value
        output.attrs["schema_version"] = SCHEMA_VERSION
        output.attrs["source_kind"] = "temperature-grid-interpolation"
        output.attrs["source_snapshot"] = str(source_path)
        output.attrs["temperature_spacing_k"] = temperature_spacing
        output.attrs["include_original"] = include_original
        output.attrs["augmentation_summary"] = json.dumps(
            summary, sort_keys=True
        )
        output.create_dataset(
            "species_names",
            data=np.asarray(species_names, dtype=object),
            dtype=string_dtype,
        )
        cells = output.create_group("cells")
        cells.create_dataset(
            "states", data=output_states, compression="gzip"
        )
        cells.create_dataset(
            "coordinates", data=output_coordinates, compression="gzip"
        )
        cells.create_dataset(
            "source_cell_index",
            data=output_source_index,
            compression="gzip",
        )
        cells.create_dataset(
            "parent_source_cell_index",
            data=output_parents,
            compression="gzip",
        )
        cells.create_dataset(
            "interpolation_fraction",
            data=output_fractions,
            compression="gzip",
        )
        cells.create_dataset(
            "is_original", data=original_mask, compression="gzip"
        )
        cells.create_dataset(
            "stratum",
            data=np.full(output_states.shape[0], -1, dtype=np.int32),
            compression="gzip",
        )
        cells.create_dataset(
            "sampling_probability",
            data=np.ones(output_states.shape[0], dtype=np.float64),
            compression="gzip",
        )
        cells.create_dataset(
            "interpolation_temperature",
            data=np.concatenate(
                [
                    states[:, 0]
                    if include_original
                    else np.empty(0, dtype=np.float64),
                    generated_temperatures,
                ]
            ),
            compression="gzip",
        )
    return summary


def split_interpolated_snapshot_spatial_blocks(
    source_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    *,
    axial_blocks: int,
    radial_blocks: int,
    validation_fraction: float,
    seed: int,
) -> dict:
    if axial_blocks < 2 or radial_blocks < 2:
        raise ValueError("spatial block counts must be at least two")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    source_path = Path(source_path)
    with h5py.File(source_path, "r") as source:
        cells = source["cells"]
        arrays = {
            name: np.asarray(dataset)
            for name, dataset in cells.items()
        }
        species_names = _decode(source["species_names"][:])
        attrs = dict(source.attrs)

    original = arrays["is_original"].astype(bool)
    original_indexes = np.flatnonzero(original)
    original_coordinates = arrays["coordinates"][original_indexes]
    source_ids = arrays["source_cell_index"][original_indexes]
    if np.any(source_ids < 0):
        raise ValueError("original states require nonnegative source cell IDs")

    lower = original_coordinates.min(axis=0)
    span = np.maximum(original_coordinates.max(axis=0) - lower, 1e-300)
    normalized = (original_coordinates - lower) / span
    axial = np.minimum(
        (normalized[:, 0] * axial_blocks).astype(np.int64),
        axial_blocks - 1,
    )
    radial = np.minimum(
        (normalized[:, 1] * radial_blocks).astype(np.int64),
        radial_blocks - 1,
    )
    block = axial * radial_blocks + radial
    occupied_blocks = np.unique(block)
    rng = np.random.default_rng(seed)
    parents = arrays["parent_source_cell_index"]
    synthetic = ~original
    active_source_ids = set(
        int(value) for value in np.unique(parents[synthetic])
    )
    source_block = {
        int(source_id): int(block_id)
        for source_id, block_id in zip(source_ids, block)
    }
    active_original = np.asarray(
        [int(value) in active_source_ids for value in source_ids],
        dtype=bool,
    )
    best_score = float("inf")
    validation_blocks = np.empty(0, dtype=np.int64)
    for _ in range(5000):
        selected = occupied_blocks[
            rng.random(occupied_blocks.size) < validation_fraction
        ]
        if selected.size == 0 or selected.size == occupied_blocks.size:
            continue
        validation_trial = np.isin(block, selected)
        original_fraction = float(np.mean(validation_trial))
        active_fraction = (
            float(np.mean(validation_trial[active_original]))
            if np.any(active_original)
            else original_fraction
        )
        selected_set = set(int(value) for value in selected)
        retained_synthetic = np.asarray(
            [
                source_block.get(int(first)) not in selected_set
                and source_block.get(int(second)) not in selected_set
                for first, second in parents[synthetic]
            ],
            dtype=bool,
        )
        retained_fraction = float(np.mean(retained_synthetic))
        score = (
            abs(original_fraction - validation_fraction)
            + abs(active_fraction - validation_fraction)
            + abs(retained_fraction - (1.0 - validation_fraction))
        )
        if score < best_score:
            best_score = score
            validation_blocks = np.sort(selected)
    if validation_blocks.size == 0:
        raise RuntimeError("failed to construct a spatial block split")
    original_validation = np.isin(block, validation_blocks)
    validation_source_ids = set(
        int(value) for value in source_ids[original_validation]
    )
    training_source_ids = set(
        int(value) for value in source_ids[~original_validation]
    )

    synthetic_train = synthetic & np.asarray(
        [
            int(first) in training_source_ids
            and int(second) in training_source_ids
            for first, second in parents
        ],
        dtype=bool,
    )
    train_mask = np.zeros(original.shape[0], dtype=bool)
    train_mask[original_indexes[~original_validation]] = True
    train_mask |= synthetic_train
    validation_mask = np.zeros(original.shape[0], dtype=bool)
    validation_mask[original_indexes[original_validation]] = True

    def write(path: Path, mask: np.ndarray, split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as output:
            for name, value in attrs.items():
                output.attrs[name] = value
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["split"] = split
            output.attrs["spatial_split_seed"] = seed
            output.attrs["axial_blocks"] = axial_blocks
            output.attrs["radial_blocks"] = radial_blocks
            output.attrs["validation_blocks"] = json.dumps(
                validation_blocks.tolist()
            )
            output.create_dataset(
                "species_names",
                data=np.asarray(species_names, dtype=object),
                dtype=string_dtype,
            )
            cells = output.create_group("cells")
            for name, values in arrays.items():
                if values.shape[0] == mask.shape[0]:
                    cells.create_dataset(
                        name, data=values[mask], compression="gzip"
                    )

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    write(train_path, train_mask, "train")
    write(validation_path, validation_mask, "validation")
    return {
        "source": str(source_path),
        "train_output": str(train_path),
        "validation_output": str(validation_path),
        "axial_blocks": axial_blocks,
        "radial_blocks": radial_blocks,
        "occupied_blocks": int(occupied_blocks.size),
        "validation_blocks": validation_blocks.tolist(),
        "split_objective": best_score,
        "validation_original_fraction": float(
            np.count_nonzero(validation_mask) / original_indexes.size
        ),
        "validation_temperature_active_fraction": float(
            np.mean(
                np.isin(block[active_original], validation_blocks)
            )
        ),
        "train_original_states": int(np.count_nonzero(train_mask & original)),
        "train_synthetic_states": int(
            np.count_nonzero(train_mask & synthetic)
        ),
        "validation_original_states": int(
            np.count_nonzero(validation_mask)
        ),
        "discarded_cross_split_synthetic_states": int(
            np.count_nonzero(synthetic & ~synthetic_train)
        ),
    }


def split_snapshot_random(
    source_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    *,
    validation_fraction: float,
    seed: int,
) -> dict:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    source_path = Path(source_path)
    with h5py.File(source_path, "r") as source:
        cells = source["cells"]
        arrays = {
            name: np.asarray(dataset)
            for name, dataset in cells.items()
        }
        species_names = _decode(source["species_names"][:])
        attrs = dict(source.attrs)
    count = arrays["states"].shape[0]
    rng = np.random.default_rng(seed)
    indexes = rng.permutation(count)
    validation_count = int(round(validation_fraction * count))
    validation_indexes = np.sort(indexes[:validation_count])
    train_indexes = np.sort(indexes[validation_count:])

    def write(path: Path, selected: np.ndarray, split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as output:
            for name, value in attrs.items():
                output.attrs[name] = value
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["split"] = split
            output.attrs["split_mode"] = "random"
            output.attrs["split_seed"] = seed
            output.create_dataset(
                "species_names",
                data=np.asarray(species_names, dtype=object),
                dtype=string_dtype,
            )
            output_cells = output.create_group("cells")
            for name, values in arrays.items():
                if values.shape[0] == count:
                    output_cells.create_dataset(
                        name, data=values[selected], compression="gzip"
                    )

    train_path = Path(train_path)
    validation_path = Path(validation_path)
    write(train_path, train_indexes, "train")
    write(validation_path, validation_indexes, "validation")
    is_original = arrays.get(
        "is_original", np.ones(count, dtype=np.uint8)
    ).astype(bool)
    return {
        "source": str(source_path),
        "train_output": str(train_path),
        "validation_output": str(validation_path),
        "mode": "random",
        "seed": seed,
        "train_states": int(train_indexes.size),
        "validation_states": int(validation_indexes.size),
        "train_original_states": int(
            np.count_nonzero(is_original[train_indexes])
        ),
        "train_synthetic_states": int(
            np.count_nonzero(~is_original[train_indexes])
        ),
        "validation_original_states": int(
            np.count_nonzero(is_original[validation_indexes])
        ),
        "validation_synthetic_states": int(
            np.count_nonzero(~is_original[validation_indexes])
        ),
    }
