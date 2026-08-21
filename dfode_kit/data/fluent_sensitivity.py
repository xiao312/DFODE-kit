from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


FLUENT_DI_JVP_SCHEMA_VERSION = "fluent-di-jvp-v1"
FLUENT_NATIVE_DI_BACKEND = "ansys-fluent-native-di"
_FORBIDDEN_PROVENANCE_TOKENS = ("cantera", "cvode", "sundials")


def validate_fluent_native_provenance(provenance: Mapping[str, Any]) -> None:
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping")
    backend = str(provenance.get("label_backend", "")).strip().lower()
    if backend != FLUENT_NATIVE_DI_BACKEND:
        raise ValueError(
            "Fluent-bound sensitivity data requires label_backend="
            f"{FLUENT_NATIVE_DI_BACKEND!r}"
        )
    if "exact_dt_seconds" not in provenance:
        raise ValueError("Fluent DI provenance requires exact_dt_seconds")
    try:
        exact_dt = float(provenance["exact_dt_seconds"])
    except (TypeError, ValueError) as error:
        raise ValueError("exact_dt_seconds must be a positive finite scalar") from error
    if not np.isfinite(exact_dt) or exact_dt <= 0.0:
        raise ValueError("exact_dt_seconds must be a positive finite scalar")
    serialized = json.dumps(
        provenance, sort_keys=True, default=_json_default
    ).lower()
    forbidden = [
        token for token in _FORBIDDEN_PROVENANCE_TOKENS if token in serialized
    ]
    if forbidden:
        raise ValueError(
            "Fluent-bound sensitivity data cannot use Cantera/CVODE/SUNDIALS "
            f"provenance; found {forbidden}"
        )


def finite_difference_jvp(
    phi_di_x,
    phi_di_x_perturbed,
    epsilon,
) -> np.ndarray:
    base = np.asarray(phi_di_x, dtype=np.float64)
    perturbed = np.asarray(phi_di_x_perturbed, dtype=np.float64)
    if base.shape != perturbed.shape or base.ndim != 2:
        raise ValueError("DI endpoints must share shape (n_samples, n_outputs)")
    eps = _sample_vector(epsilon, base.shape[0], name="epsilon")
    if np.any(eps <= 0.0):
        raise ValueError("epsilon must be positive")
    return (perturbed - base) / eps[:, None]


@dataclass
class FluentDIJVPBatch:
    x: np.ndarray
    x_perturbed: np.ndarray
    phi_di_x: np.ndarray
    phi_di_x_perturbed: np.ndarray
    epsilon: np.ndarray
    direction: np.ndarray
    dt: np.ndarray
    species_names: Sequence[str]
    provenance: Mapping[str, Any]
    sample_metadata: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.x_perturbed = np.asarray(self.x_perturbed, dtype=np.float64)
        self.phi_di_x = np.asarray(self.phi_di_x, dtype=np.float64)
        self.phi_di_x_perturbed = np.asarray(
            self.phi_di_x_perturbed, dtype=np.float64
        )
        if self.x.ndim != 2:
            raise ValueError("x must have shape (n_samples, 2 + n_species)")
        n_samples = self.x.shape[0]
        self.epsilon = _sample_vector(
            self.epsilon, n_samples, name="epsilon"
        )
        self.dt = _sample_vector(self.dt, n_samples, name="dt")
        self.direction = np.asarray(self.direction, dtype=np.float64)
        self.species_names = tuple(str(name) for name in self.species_names)
        self.provenance = dict(self.provenance)
        self.sample_metadata = {
            str(name): np.asarray(values)
            for name, values in self.sample_metadata.items()
        }
        self.validate()

    @property
    def jvp_target(self) -> np.ndarray:
        return finite_difference_jvp(
            self.phi_di_x, self.phi_di_x_perturbed, self.epsilon
        )

    def validate(self) -> None:
        validate_fluent_native_provenance(self.provenance)
        arrays = {
            "x": self.x,
            "x_perturbed": self.x_perturbed,
            "phi_di_x": self.phi_di_x,
            "phi_di_x_perturbed": self.phi_di_x_perturbed,
            "direction": self.direction,
        }
        for name, values in arrays.items():
            if values.shape != self.x.shape:
                raise ValueError(
                    f"{name} has shape {values.shape}; expected {self.x.shape}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.epsilon <= 0.0):
            raise ValueError("epsilon must be positive")
        if np.any(self.dt <= 0.0):
            raise ValueError("dt must be positive")
        exact_dt = np.float64(self.provenance["exact_dt_seconds"])
        if not np.all(self.dt == exact_dt):
            raise ValueError(
                "every pair dt must exactly match provenance exact_dt_seconds"
            )
        if len(self.species_names) != self.x.shape[1] - 2:
            raise ValueError("species_names does not match the T,p,Y state width")
        if len(set(self.species_names)) != len(self.species_names):
            raise ValueError("species_names contains duplicates")
        if np.any(np.linalg.norm(self.direction, axis=1) == 0.0):
            raise ValueError("every perturbation direction must be nonzero")

        expected = self.x + self.epsilon[:, None] * self.direction
        if not np.allclose(
            self.x_perturbed, expected, rtol=5e-13, atol=1e-15
        ):
            max_error = float(np.max(np.abs(self.x_perturbed - expected)))
            raise ValueError(
                "x_perturbed must equal x + epsilon * direction; "
                f"max error is {max_error:.6e}"
            )
        for name, values in self.sample_metadata.items():
            if not name or "/" in name:
                raise ValueError(f"invalid sample metadata name: {name!r}")
            if values.ndim == 0 or values.shape[0] != self.x.shape[0]:
                raise ValueError(
                    f"sample metadata {name!r} must have one row per sample"
                )


def write_fluent_di_jvp_dataset(
    path: str | Path,
    batch: FluentDIJVPBatch,
) -> dict[str, Any]:
    batch.validate()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    provenance_json = json.dumps(
        batch.provenance, sort_keys=True, default=_json_default
    )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = FLUENT_DI_JVP_SCHEMA_VERSION
        handle.attrs["label_backend"] = FLUENT_NATIVE_DI_BACKEND
        handle.attrs["exact_dt_seconds"] = float(
            batch.provenance["exact_dt_seconds"]
        )
        handle.attrs["provenance_json"] = provenance_json
        handle.attrs["n_pairs"] = int(batch.x.shape[0])
        handle.attrs["state_layout"] = "T,p,Y[species_names]"
        handle.attrs["perturbation_relation"] = (
            "x_perturbed=x+epsilon*direction"
        )
        handle.create_dataset(
            "species_names",
            data=np.asarray(batch.species_names, dtype=object),
            dtype=string_dtype,
        )
        pairs = handle.create_group("pairs")
        for name, values in (
            ("x", batch.x),
            ("x_perturbed", batch.x_perturbed),
            ("phi_di_x", batch.phi_di_x),
            ("phi_di_x_perturbed", batch.phi_di_x_perturbed),
            ("epsilon", batch.epsilon),
            ("direction", batch.direction),
            ("dt", batch.dt),
            ("jvp_target", batch.jvp_target),
        ):
            pairs.create_dataset(name, data=values, compression="gzip")

        metadata = handle.create_group("sample_metadata")
        for name, values in batch.sample_metadata.items():
            if values.dtype.kind in {"O", "S", "U"}:
                text_values = np.asarray(values, dtype=str).astype(object)
                metadata.create_dataset(
                    name, data=text_values, dtype=string_dtype
                )
            else:
                metadata.create_dataset(name, data=values, compression="gzip")

    return {
        "output": str(output_path),
        "schema_version": FLUENT_DI_JVP_SCHEMA_VERSION,
        "label_backend": FLUENT_NATIVE_DI_BACKEND,
        "n_pairs": int(batch.x.shape[0]),
        "state_width": int(batch.x.shape[1]),
        "epsilon_min": float(np.min(batch.epsilon)),
        "epsilon_max": float(np.max(batch.epsilon)),
    }


def load_fluent_di_jvp_dataset(path: str | Path) -> FluentDIJVPBatch:
    with h5py.File(path, "r") as handle:
        schema = _decode_scalar(handle.attrs.get("schema_version", ""))
        if schema != FLUENT_DI_JVP_SCHEMA_VERSION:
            raise ValueError(
                f"expected {FLUENT_DI_JVP_SCHEMA_VERSION}, got {schema!r}"
            )
        provenance = json.loads(
            _decode_scalar(handle.attrs.get("provenance_json", "{}"))
        )
        pairs = handle["pairs"]
        stored_jvp = np.asarray(pairs["jvp_target"], dtype=np.float64)
        metadata = {
            name: _decode_string_array(dataset[:])
            for name, dataset in handle.get("sample_metadata", {}).items()
        }
        batch = FluentDIJVPBatch(
            x=np.asarray(pairs["x"], dtype=np.float64),
            x_perturbed=np.asarray(
                pairs["x_perturbed"], dtype=np.float64
            ),
            phi_di_x=np.asarray(pairs["phi_di_x"], dtype=np.float64),
            phi_di_x_perturbed=np.asarray(
                pairs["phi_di_x_perturbed"], dtype=np.float64
            ),
            epsilon=np.asarray(pairs["epsilon"], dtype=np.float64),
            direction=np.asarray(pairs["direction"], dtype=np.float64),
            dt=np.asarray(pairs["dt"], dtype=np.float64),
            species_names=[
                _decode_scalar(value) for value in handle["species_names"][:]
            ],
            provenance=provenance,
            sample_metadata=metadata,
        )
    if not np.allclose(
        stored_jvp, batch.jvp_target, rtol=1e-13, atol=0.0, equal_nan=False
    ):
        raise ValueError("stored jvp_target does not match paired DI endpoints")
    return batch


def deterministic_jvp_split_indices(
    n_pairs: int,
    *,
    validation_fraction: float = 0.2,
    seed: int = 260624,
) -> tuple[np.ndarray, np.ndarray]:
    """Split pair IDs reproducibly without inspecting chemistry targets."""

    if n_pairs < 2:
        raise ValueError("a JVP train/validation split requires at least two pairs")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    generator = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
    order = generator.permutation(n_pairs).astype(np.int64, copy=False)
    validation_count = min(
        n_pairs - 1,
        max(1, int(round(validation_fraction * n_pairs))),
    )
    return order[validation_count:].copy(), order[:validation_count].copy()


def deterministic_paired_batch_indices(
    pair_indices,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    rank: int = 0,
    world_size: int = 1,
    shuffle: bool = True,
) -> tuple[np.ndarray, ...]:
    """Return deterministic, pair-preserving batches for one DDP rank.

    The shuffled global order is truncated, rather than padded, when necessary
    to give every rank the same number of pairs. Consequently no pair is
    duplicated or posterior-resampled within an epoch.
    """

    indices = np.asarray(pair_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("pair_indices cannot be empty")
    if np.unique(indices).size != indices.size:
        raise ValueError("pair_indices must not contain duplicates")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")

    order = indices.copy()
    if shuffle:
        seed_sequence = np.random.SeedSequence(
            [int(seed) & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF]
        )
        order = np.random.default_rng(seed_sequence).permutation(order)
    if world_size > 1:
        usable = (order.size // world_size) * world_size
        if usable == 0:
            raise ValueError("JVP partition has fewer pairs than DDP ranks")
        order = order[:usable].reshape(-1, world_size)[:, rank]
    return tuple(
        order[start : start + batch_size].copy()
        for start in range(0, order.size, batch_size)
    )


def _sample_vector(values, n_samples: int, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(n_samples, float(result), dtype=np.float64)
    else:
        result = result.reshape(-1)
    if result.shape[0] != n_samples:
        raise ValueError(f"{name} must have one value per sample")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _decode_scalar(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind not in {"O", "S", "U"}:
        return values
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ]
    )


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize provenance value {value!r}")
