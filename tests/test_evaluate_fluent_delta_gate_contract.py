from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_fluent_delta_gate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_fluent_delta_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_total_enthalpy_contract_is_not_interpreted_as_temperature() -> None:
    output = np.arange(12, dtype=np.float64).reshape(3, 4)
    manifest = {
        "thermochemical_closure": {
            "predicts_temperature": False,
            "predicts_total_enthalpy_increment": True,
        }
    }
    delta_t, delta_h, delta_y = MODULE._split_model_output(output, 3, manifest)
    assert delta_t is None
    np.testing.assert_array_equal(delta_h, output[:, 0])
    np.testing.assert_array_equal(delta_y, output[:, 1:])


def test_temperature_and_legacy_contracts_remain_supported() -> None:
    output = np.arange(8, dtype=np.float64).reshape(2, 4)
    temperature_manifest = {
        "thermochemical_closure": {
            "predicts_temperature": True,
            "predicts_total_enthalpy_increment": False,
        }
    }
    for manifest in (temperature_manifest, {}):
        delta_t, delta_h, delta_y = MODULE._split_model_output(
            output, 3, manifest
        )
        np.testing.assert_array_equal(delta_t, output[:, 0])
        assert delta_h is None
        np.testing.assert_array_equal(delta_y, output[:, 1:])


def test_manifest_and_output_width_must_agree() -> None:
    manifest = {
        "thermochemical_closure": {
            "predicts_temperature": False,
            "predicts_total_enthalpy_increment": True,
        }
    }
    with pytest.raises(ValueError, match="missing thermochemical"):
        MODULE._split_model_output(np.zeros((2, 3)), 3, manifest)
    contradictory = {
        "thermochemical_closure": {
            "predicts_temperature": True,
            "predicts_total_enthalpy_increment": True,
        }
    }
    with pytest.raises(ValueError, match="cannot predict both"):
        MODULE._split_model_output(np.zeros((2, 4)), 3, contradictory)
