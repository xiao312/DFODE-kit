from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_fluent_deployment_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_fluent_deployment_metrics", SCRIPT)
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
        consumed = -2.0 * current_species[:, :1]
        delta_y = torch.cat((consumed, -consumed), dim=1)
        return torch.cat((delta_temperature, delta_y), dim=1)


def write_pairs(path: Path, *, backend: str) -> None:
    current = np.asarray(
        [[1000.0, 101325.0, 0.2, 0.8], [1200.0, 101325.0, 0.1, 0.9]],
        dtype=np.float64,
    )
    target = current.copy()
    target[:, 0] += 1.0
    target[:, 2] -= 0.05
    target[:, 3] += 0.05
    with h5py.File(path, "w") as handle:
        handle.attrs["label_backend"] = backend
        handle.attrs["species_name_contract"] = (
            "cantera-canonical-with-fluent-aliases"
        )
        handle.create_dataset(
            "species_names",
            data=np.asarray(["A", "B"], dtype=h5py.string_dtype("utf-8")),
        )
        pairs = handle.create_group("pairs")
        pairs.create_dataset("current_states", data=current)
        pairs.create_dataset("target_states", data=target)
        pairs.create_dataset("dt", data=np.asarray([1.0e-6, 1.0e-6]))
        pairs.create_dataset("source_index", data=np.asarray([4, 9], dtype=np.uint64))
        pairs.create_dataset("mixture_fraction", data=np.asarray([0.1, 0.8]))


def write_artifact(path: Path) -> None:
    path.mkdir()
    module = FakeArtifact().eval()
    traced = torch.jit.trace(
        module,
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
                "output": {
                    "meaning": "delta_T followed by positive delta_Y",
                    "shape": ["batch", 3],
                },
                "hard_layer": {"current_species_source": "separate_input"},
                "training_config": {"variant": "neural-patankar"},
            }
        ),
        encoding="utf-8",
    )


def test_rejects_non_fluent_label_provenance(tmp_path: Path) -> None:
    source = tmp_path / "pairs.h5"
    write_pairs(source, backend="cantera-cvode")
    with pytest.raises(ValueError, match="rejects Cantera/CVODE/SUNDIALS"):
        MODULE.load_fluent_native_pairs(source)


def test_hard_limiter_matches_fluent_udf_and_reports_controller() -> None:
    current = np.asarray([[0.2, 0.8]], dtype=np.float64)
    delta = np.asarray([[-0.4, 0.4]], dtype=np.float64)
    limited, scale, controlling = MODULE.apply_deployment_hard_limiter(
        current, delta
    )
    assert scale[0] == pytest.approx(0.999999 * 0.2 / 0.4)
    assert controlling.tolist() == [0]
    assert np.min(current + limited) > 0.0
    assert np.sum(limited, axis=1)[0] == pytest.approx(0.0, abs=1.0e-16)


def test_density_contracts_separate_label_pressure_from_chemistry_pressure() -> None:
    contracts = MODULE._density_contract_metrics(
        density_current=np.asarray([1.0], dtype=np.float64),
        density_target_labeled_state=np.asarray([0.90], dtype=np.float64),
        density_target_fixed_pressure=np.asarray([0.98], dtype=np.float64),
        density_prediction_fixed_pressure=np.asarray([0.98], dtype=np.float64),
        pressure_current=np.asarray([101325.0], dtype=np.float64),
        pressure_target=np.asarray([101025.0], dtype=np.float64),
    )

    full = contracts["full_labeled_state_density"]
    fixed = contracts[
        "deployment_consistent_fixed_pressure_chemistry_density"
    ]
    assert full["controllable_by_chemistry_artifact"] is False
    assert full["labeled_pressure_change_pa"]["mean"] == pytest.approx(-300.0)
    assert full["density_increment"]["nmae"] == pytest.approx(0.8)
    assert fixed["controllable_by_chemistry_artifact"] is True
    assert fixed["density_increment"]["nmae"] == pytest.approx(0.0)


def test_legacy_density_metric_is_explicitly_deprecated() -> None:
    contracts = MODULE._density_contract_metrics(
        density_current=np.asarray([1.0], dtype=np.float64),
        density_target_labeled_state=np.asarray([0.90], dtype=np.float64),
        density_target_fixed_pressure=np.asarray([0.98], dtype=np.float64),
        density_prediction_fixed_pressure=np.asarray([0.98], dtype=np.float64),
        pressure_current=np.asarray([101325.0], dtype=np.float64),
        pressure_target=np.asarray([101025.0], dtype=np.float64),
    )
    legacy = MODULE._deprecated_density_increment_alias(contracts)

    assert legacy["deprecated"] is True
    assert legacy["alias_of"] == (
        "density_contracts.full_labeled_state_density.density_increment"
    )
    assert legacy["nmae"] == contracts["full_labeled_state_density"][
        "density_increment"
    ]["nmae"]


def test_end_to_end_artifact_evaluation_without_offline_thermo(
    tmp_path: Path,
) -> None:
    source = tmp_path / "pairs.h5"
    artifact = tmp_path / "artifact"
    write_pairs(source, backend="fluent-native-direct-integration")
    write_artifact(artifact)
    parser = MODULE.build_parser()
    args = parser.parse_args(
        [
            "--artifact-dir",
            str(artifact),
            "--source",
            str(source),
            "--output",
            str(tmp_path / "report.json"),
            "--device",
            "cpu",
            "--thermo",
            "none",
            "--mixture-fraction-key",
            "mixture_fraction",
        ]
    )
    report = MODULE.run_evaluation(args)

    assert report["schema_version"] == 2
    assert report["provenance"]["normalized_backend"] == "ansys-fluent-native-di"
    assert report["artifact_contract"]["predicts_pressure"] is False
    assert report["sampling"]["evaluated_rows"] == 2
    assert report["stage_semantics"]["raw_equals_post_model_for_exported_artifact"]
    assert report["limiter"]["activation_rate"] == pytest.approx(1.0)
    assert report["limiter"]["controlling_species"][0]["species"] == "A"
    assert report["endpoint_metrics"]["raw_model_endpoint"][
        "negative_cell_rate"
    ] == pytest.approx(1.0)
    assert report["endpoint_metrics"]["deployment_hard_limiter_endpoint"][
        "negative_cell_rate"
    ] == pytest.approx(0.0)
    assert "temperature" in report["delta_y_over_dt"]["deployment_by_condition"]
    assert "dt" in report["delta_y_over_dt"]["deployment_by_condition"]
    assert "mixture_fraction" in report["delta_y_over_dt"][
        "deployment_by_condition"
    ]
    assert report["offline_thermodynamics"]["available"] is False
