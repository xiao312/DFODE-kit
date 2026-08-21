from __future__ import annotations

import argparse

import h5py
import numpy as np
import pytest
import torch

from dfode_kit.cli.commands.train_positive_interval import add_command_parser
from dfode_kit.models.positive_interval import (
    DELTA_H_TOTAL_MODE,
    DELTA_TEMPERATURE_MODE,
    NeuralPatankarIntervalModel,
)
from dfode_kit.training import positive_interval as training


REACTANTS = np.asarray([[1.0], [0.0]], dtype=np.float64)
PRODUCTS = np.asarray([[0.0], [1.0]], dtype=np.float64)
REVERSIBLE = np.asarray([False])


def _model(mode: str) -> NeuralPatankarIntervalModel:
    return NeuralPatankarIntervalModel(
        input_dim=4,
        reactant_mass_matrix=REACTANTS,
        product_mass_matrix=PRODUCTS,
        reaction_reversible=REVERSIBLE,
        latent_dim=4,
        hidden_dim=8,
        extent_scale=1.0e-3,
        thermochemical_output_mode=mode,
        total_enthalpy_delta_scale=500.0,
    )


def test_legacy_temperature_mode_preserves_head_and_float64_physical_species():
    model = _model(DELTA_TEMPERATURE_MODE)
    output = model(
        torch.zeros((3, 4), dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.float32),
        current_species=torch.tensor(
            [[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]], dtype=torch.float64
        ),
    )

    assert hasattr(model, "temperature_delta_head")
    assert not hasattr(model, "total_enthalpy_delta_head")
    assert output["delta_temperature"].dtype == torch.float64
    assert output["delta_species"].dtype == torch.float64
    assert output["next_species"].dtype == torch.float64
    assert "delta_h_total" not in output


def test_total_enthalpy_mode_has_no_temperature_head_and_keeps_hard_layer_float64():
    model = _model(DELTA_H_TOTAL_MODE)
    current_y = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    output = model(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
        current_species=current_y,
    )

    assert hasattr(model, "total_enthalpy_delta_head")
    assert not hasattr(model, "temperature_delta_head")
    assert not hasattr(model, "tp_head")
    assert "delta_temperature" not in output
    assert output["reaction_extent"].dtype == torch.float64
    assert output["delta_species"].dtype == torch.float64
    assert output["next_species"].dtype == torch.float64
    assert output["delta_h_total"].dtype == torch.float64
    torch.testing.assert_close(
        output["next_species"],
        current_y + output["delta_species"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output["delta_species"].sum(dim=1),
        torch.zeros(1, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )
    assert torch.all(output["next_species"] >= 0.0)


def test_total_enthalpy_label_loader_requires_both_fluent_fields(tmp_path):
    path = tmp_path / "labels.h5"
    with h5py.File(path, "w") as h5:
        pairs = h5.create_group("pairs")
        pairs.create_dataset("h_total_before", data=[1.0, 2.0])

    with pytest.raises(ValueError, match="pairs/h_total_after"):
        training._load_total_enthalpy_increment_labels(
            str(path), expected_rows=2
        )

    with h5py.File(path, "a") as h5:
        h5["pairs"].create_dataset("h_total_after", data=[1.5, 1.0])
    before, after, delta = training._load_total_enthalpy_increment_labels(
        str(path), expected_rows=2
    )
    assert before.dtype == np.float64
    assert after.dtype == np.float64
    np.testing.assert_array_equal(delta, np.asarray([0.5, -1.0]))


def test_total_enthalpy_mode_requires_native_fluent_labels():
    config = training.PositiveIntervalTrainingConfig(
        thermochemical_output_mode=DELTA_H_TOTAL_MODE
    )
    native = {"label_backend": "fluent-native-direct-integration"}
    assert training._require_deployment_label_sources(config, native, native) == (
        "ansys-fluent-native-di",
        "ansys-fluent-native-di",
    )
    with pytest.raises(ValueError, match="Fluent native direct integration"):
        training._require_deployment_label_sources(
            config,
            {"label_backend": "cantera-cvode"},
            native,
        )


def test_cli_defaults_remain_legacy_and_accept_total_enthalpy_mode():
    parser = argparse.ArgumentParser()
    add_command_parser(parser.add_subparsers(dest="command"))
    required = [
        "train-positive-interval",
        "--train-source",
        "train.h5",
        "--validation-source",
        "validation.h5",
        "--mech",
        "gri30.yaml",
        "--output",
        "model.pt",
        "--variant",
        "neural-patankar",
    ]
    legacy = parser.parse_args(required)
    total_h = parser.parse_args(
        required
        + [
            "--thermochemical-output-mode",
            "delta-h-total",
            "--total-enthalpy-delta-scale",
            "250",
            "--total-enthalpy-delta-loss-weight",
            "0.5",
        ]
    )

    assert legacy.thermochemical_output_mode == "delta-temperature"
    assert total_h.thermochemical_output_mode == "delta-h-total"
    assert total_h.total_enthalpy_delta_scale == 250.0
    assert total_h.total_enthalpy_delta_loss_weight == 0.5
