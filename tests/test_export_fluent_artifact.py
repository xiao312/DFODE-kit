from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts import export_fluent_artifact as exporter


class _FakePatankarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(6, 4)
        self.demand_head = torch.nn.Linear(4, 2)
        self.temperature_delta_head = torch.nn.Linear(7, 1)
        self.temperature_delta_scale = 10.0
        self.total_enthalpy_delta_head = torch.nn.Linear(7, 1)
        self.total_enthalpy_delta_scale = 1000.0
        self.extent_scale = 1.0e-2
        self.register_buffer(
            "consumption",
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "process_stoich",
            torch.tensor(
                [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
                dtype=torch.float64,
            ),
        )
        torch.manual_seed(7)
        for parameter in self.parameters():
            torch.nn.init.uniform_(parameter, -0.1, 0.1)


def _export(tmp_path, monkeypatch, closure=None, model_mode=None):
    checkpoint_path = tmp_path / "checkpoint.pt"
    mechanism_path = tmp_path / "mechanism.yaml"
    output_dir = tmp_path / (closure or "default")
    checkpoint = {
        "model_type": "stoichiometric_interval",
        "positive_model_type": "neural-patankar",
        "state_mean": np.asarray(
            [1000.0, 101325.0, 0.4, 0.3, 0.3], dtype=np.float64
        ),
        "state_std": np.asarray(
            [100.0, 1000.0, 0.1, 0.1, 0.1], dtype=np.float64
        ),
        "log_dt_mean": np.asarray([np.log(1.0e-6)], dtype=np.float64),
        "log_dt_std": np.asarray([0.5], dtype=np.float64),
        "species_names": ["A", "B", "C"],
        "molecular_weights": np.ones(3, dtype=np.float64),
        "stoichiometric_matrix": np.asarray(
            [[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        "training_config": {
            "transform_alpha": 0.1,
            "thermochemical_output_mode": model_mode or "delta-temperature",
            "total_enthalpy_delta_scale": 1000.0,
        },
        "phase_name": "gas",
    }
    if model_mode is not None:
        checkpoint["thermochemical_output_mode"] = model_mode
        checkpoint["total_enthalpy_delta_scale"] = 1000.0
        checkpoint["thermochemical_target_contract"] = {
            "before_dataset": "pairs/h_total_before",
            "after_dataset": "pairs/h_total_after",
        }
    torch.save(checkpoint, checkpoint_path)
    mechanism_path.write_text("synthetic mechanism\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "cantera",
        SimpleNamespace(Solution=lambda _: object()),
    )
    model = _FakePatankarModel().eval()
    monkeypatch.setattr(
        exporter,
        "_load_export_positive_model",
        lambda checkpoint, gas, state_width, device: model,
    )
    kwargs = {} if closure is None else {"thermochemical_closure": closure}
    manifest = exporter.export_artifact(
        checkpoint_path,
        mechanism_path,
        output_dir,
        **kwargs,
    )
    return manifest, output_dir


def _runtime_config(path):
    return dict(
        line.split("=", maxsplit=1)
        for line in path.read_text(encoding="ascii").splitlines()
    )


def _assert_reference_replays(output_dir, expected_width):
    reference = np.load(output_dir / "reference_io.npz")
    traced = torch.jit.load(str(output_dir / "model.pt"))
    with torch.inference_mode():
        actual = traced(
            torch.from_numpy(reference["physical_input"]),
            torch.from_numpy(reference["current_species"]),
        ).numpy()
    np.testing.assert_allclose(actual, reference["model_output"])
    assert actual.shape == (4, expected_width)
    assert reference["model_output_fields"].shape == (expected_width,)


def test_default_export_preserves_predicted_delta_temperature(
    tmp_path,
    monkeypatch,
):
    manifest, output_dir = _export(tmp_path, monkeypatch)
    runtime = _runtime_config(output_dir / "runtime.cfg")

    assert runtime["output_mode"] == "direct_delta_t_delta_y"
    assert runtime["reaction_width"] == "4"
    assert runtime["predicts_temperature"] == "1"
    assert manifest["output"]["shape"] == ["batch", 4]
    assert manifest["output"]["fields"] == [
        "delta_T",
        "delta_Y:A",
        "delta_Y:B",
        "delta_Y:C",
    ]
    assert manifest["thermochemical_closure"] == {
        "mode": exporter.PREDICTED_DELTA_T_CLOSURE,
        "predicts_temperature": True,
        "runtime_contract": "artifact_info.predicts_temperature=true",
        "temperature_update": "T_next = T_current + delta_T",
        "temperature_owner": "dfode-model",
        "enthalpy_projection": False,
    }
    _assert_reference_replays(output_dir, expected_width=4)


def test_constant_enthalpy_export_emits_species_only(
    tmp_path,
    monkeypatch,
):
    manifest, output_dir = _export(
        tmp_path,
        monkeypatch,
        closure=exporter.FLUENT_CONSTANT_ENTHALPY_CLOSURE,
    )
    runtime = _runtime_config(output_dir / "runtime.cfg")

    assert runtime["output_mode"] == "direct_delta_y"
    assert runtime["reaction_width"] == "3"
    assert runtime["predicts_temperature"] == "0"
    assert manifest["artifact_type"] == "dfode-fluent-direct-delta-y"
    assert manifest["output"]["shape"] == ["batch", 3]
    assert manifest["output"]["fields"] == [
        "delta_Y:A",
        "delta_Y:B",
        "delta_Y:C",
    ]
    closure = manifest["thermochemical_closure"]
    assert closure["mode"] == exporter.FLUENT_CONSTANT_ENTHALPY_CLOSURE
    assert closure["predicts_temperature"] is False
    assert closure["runtime_contract"] == (
        "artifact_info.predicts_temperature=false"
    )
    assert closure["thermodynamics_backend"] == "ansys-fluent"
    assert closure["python_thermodynamics_equivalence_claim"] == "none"
    _assert_reference_replays(output_dir, expected_width=3)


def test_total_enthalpy_increment_export_uses_fluent_recovery_contract(
    tmp_path,
    monkeypatch,
):
    manifest, output_dir = _export(
        tmp_path,
        monkeypatch,
        model_mode="delta-h-total",
    )
    runtime = _runtime_config(output_dir / "runtime.cfg")

    assert runtime["output_mode"] == "direct_delta_h_total_delta_y"
    assert runtime["reaction_width"] == "4"
    assert runtime["predicts_temperature"] == "0"
    assert runtime["predicts_total_enthalpy_increment"] == "1"
    assert manifest["schema_version"] == 4
    assert manifest["artifact_type"] == (
        "dfode-fluent-direct-delta-h-total-delta-y"
    )
    assert manifest["output"]["fields"] == [
        "delta_h_total",
        "delta_Y:A",
        "delta_Y:B",
        "delta_Y:C",
    ]
    closure = manifest["thermochemical_closure"]
    assert closure["predicts_temperature"] is False
    assert closure["predicts_total_enthalpy_increment"] is True
    assert closure["temperature_recovery"] == (
        "T_next = Temperature(h_total_before + delta_h_total - "
        "Reference_Enthalpy(Y_next), Y_next)"
    )
    assert closure["target_contract"] == {
        "before_dataset": "pairs/h_total_before",
        "after_dataset": "pairs/h_total_after",
    }
    _assert_reference_replays(output_dir, expected_width=4)
