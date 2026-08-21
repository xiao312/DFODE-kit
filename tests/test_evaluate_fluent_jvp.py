from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

from dfode_kit.data.fluent_sensitivity import (
    FluentDIJVPBatch,
    write_fluent_di_jvp_dataset,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_fluent_jvp.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fluent_jvp", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeArtifact(torch.nn.Module):
    def forward(
        self, physical_input: torch.Tensor, current_species: torch.Tensor
    ) -> torch.Tensor:
        delta_temperature = torch.full(
            (physical_input.shape[0], 1), 2.0, dtype=torch.float64
        )
        amount = 0.1 * current_species[:, :1]
        delta_y = torch.cat((-amount, amount), dim=1)
        return torch.cat((delta_temperature, delta_y), dim=1)


def _endpoint(states: np.ndarray) -> np.ndarray:
    result = states.copy()
    result[:, 0] += 2.0
    amount = 0.1 * states[:, 2]
    result[:, 2] -= amount
    result[:, 3] += amount
    return result


def write_dataset(path: Path) -> None:
    x = np.asarray(
        [
            [900.0, 101325.0, 0.4, 0.6],
            [1100.0, 101325.0, 0.3, 0.7],
            [1400.0, 101325.0, 0.2, 0.8],
            [1800.0, 101325.0, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    direction = np.asarray(
        [
            [0.0, 0.0, -0.05, 0.05],
            [0.0, 0.0, 0.05, -0.05],
            [0.0, 0.0, -0.02, 0.02],
            [0.0, 0.0, 0.02, -0.02],
        ],
        dtype=np.float64,
    )
    epsilon = np.ones(4, dtype=np.float64)
    perturbed = x + epsilon[:, None] * direction
    batch = FluentDIJVPBatch(
        x=x,
        x_perturbed=perturbed,
        phi_di_x=_endpoint(x),
        phi_di_x_perturbed=_endpoint(perturbed),
        epsilon=epsilon,
        direction=direction,
        dt=np.full(4, 1.0e-6, dtype=np.float64),
        species_names=("A", "B"),
        provenance={
            "label_backend": "ansys-fluent-native-di",
            "exact_dt_seconds": 1.0e-6,
            "case": "synthetic-test",
        },
        sample_metadata={
            "direction_type": np.asarray(
                ["local-cfd-neighbor", "local-cfd-neighbor", "seeded-tangent", "seeded-tangent"]
            )
        },
    )
    write_fluent_di_jvp_dataset(path, batch)


def write_artifact(path: Path) -> None:
    path.mkdir()
    traced = torch.jit.trace(
        FakeArtifact().eval(),
        (
            torch.zeros((2, 5), dtype=torch.float32),
            torch.asarray([[0.2, 0.8], [0.1, 0.9]], dtype=torch.float64),
        ),
        strict=True,
    )
    traced.save(str(path / "model.pt"))
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "species_names": ["A", "B"],
                "output": {"shape": ["batch", 3]},
                "hard_layer": {"current_species_source": "separate_input"},
                "training_config": {"variant": "test"},
            }
        ),
        encoding="utf-8",
    )


def test_rejects_non_fluent_provenance(tmp_path: Path) -> None:
    source = tmp_path / "pairs.h5"
    write_dataset(source)
    with h5py.File(source, "r+") as handle:
        provenance = json.loads(handle.attrs["provenance_json"])
        provenance["label_backend"] = "cantera-cvode"
        handle.attrs["provenance_json"] = json.dumps(provenance)
    with pytest.raises(ValueError, match="Fluent-bound sensitivity data"):
        MODULE.load_fluent_di_jvp_dataset(source)


def test_limiter_switch_summary_separates_exact_and_material() -> None:
    summary = MODULE.summarize_limiter_pair_switching(
        np.asarray([1.0, 0.9999995, 0.5]),
        np.asarray([0.9999999, 0.9999994, 0.4]),
        np.asarray([-1, 2, 1]),
        np.asarray([0, 2, 1]),
        material_active_scale=0.9999,
        material_scale_change=1.0e-4,
    )
    assert summary["exact_active_status_switch_count"] == 1
    assert summary["material_active_status_switch_count"] == 0
    assert summary["material_scale_change_count"] == 1
    assert summary["controlling_species_switch_count"] == 1


def test_species_comparison_accepts_fluent_alias_syntax() -> None:
    fluent_names = ("h2", "ch2<s>", "n2")
    artifact_names = ("H2", "CH2(S)", "N2")
    assert tuple(map(MODULE.canonical_species_name, fluent_names)) == tuple(
        map(MODULE.canonical_species_name, artifact_names)
    )


def test_end_to_end_jvp_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "pairs.h5"
    artifact = tmp_path / "artifact"
    write_dataset(source)
    write_artifact(artifact)
    args = MODULE.build_parser().parse_args(
        [
            "--artifact-dir",
            str(artifact),
            "--source",
            str(source),
            "--output",
            str(tmp_path / "report.json"),
            "--device",
            "cpu",
            "--condition-bins",
            "2",
        ]
    )
    report = MODULE.run_evaluation(args)
    endpoint = report["metrics"]["endpoint_jvp"]["species"]["overall"]
    source_metrics = report["metrics"]["delta_y_over_dt_source_jvp"]["overall"]
    assert endpoint["mae"] == pytest.approx(0.0, abs=1.0e-14)
    assert source_metrics["nmae"] == pytest.approx(0.0, abs=1.0e-12)
    assert source_metrics["cosine"] == pytest.approx(1.0)
    assert len(report["condition_summaries"]["direction_type"]) == 2
    assert len(report["condition_summaries"]["temperature"]["bins"]) == 2
    assert report["limiter_pair_switching"]["material_active_status_switch_count"] == 0
