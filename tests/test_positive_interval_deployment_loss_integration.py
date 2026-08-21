from __future__ import annotations

import argparse

import pytest
import torch

from dfode_kit.cli.commands.train_positive_interval import add_command_parser
from dfode_kit.training import positive_interval as training


def _required_cli_args():
    return [
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


def test_deployment_loss_cli_defaults_are_backward_compatible_and_explicit():
    parser = argparse.ArgumentParser()
    add_command_parser(parser.add_subparsers(dest="command"))
    defaults = parser.parse_args(_required_cli_args())
    assert defaults.source_loss_weight == 0.0
    assert defaults.mixture_molecular_weight_loss_weight == 0.0
    assert defaults.density_increment_loss_weight == 0.0

    configured = parser.parse_args(
        _required_cli_args()
        + [
            "--source-loss-weight",
            "0.05",
            "--source-loss-scale",
            "0.1",
            "--mixture-molecular-weight-loss-weight",
            "0.01",
            "--mixture-molecular-weight-loss-scale",
            "0.001",
            "--density-increment-loss-weight",
            "0.02",
            "--density-increment-loss-scale",
            "0.001",
        ]
    )
    assert configured.source_loss_weight == 0.05
    assert configured.source_loss_scale == 0.1
    assert configured.mixture_molecular_weight_loss_weight == 0.01
    assert configured.mixture_molecular_weight_loss_scale == 0.001
    assert configured.density_increment_loss_weight == 0.02
    assert configured.density_increment_loss_scale == 0.001


def test_weighted_deployment_losses_require_fluent_native_di_labels():
    config = training.PositiveIntervalTrainingConfig(source_loss_weight=0.05)
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

    disabled = training.PositiveIntervalTrainingConfig()
    assert training._require_deployment_label_sources(
        disabled,
        {"label_backend": "cantera-cvode"},
        {"label_backend": "cantera-cvode"},
    ) == (None, None)


def test_batch_report_uses_fixed_scales_and_predicted_endpoint_temperature():
    config = training.PositiveIntervalTrainingConfig(
        source_loss_weight=0.05,
        source_loss_scale=0.1,
        mixture_molecular_weight_loss_weight=0.01,
        mixture_molecular_weight_loss_scale=1.0e-3,
        density_increment_loss_weight=0.02,
        density_increment_loss_scale=1.0e-3,
    )
    current_y = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    predicted_delta_y = torch.tensor(
        [[-1.0e-4, 1.0e-4]], dtype=torch.float64, requires_grad=True
    )
    target_delta_y = torch.tensor([[-0.8e-4, 0.8e-4]], dtype=torch.float64)
    report = training._deployment_loss_for_batch(
        loss_config=training._deployment_loss_config(config),
        current_y=current_y,
        predicted_delta_y=predicted_delta_y,
        target_delta_y=target_delta_y,
        current_temperature=torch.tensor([1000.0], dtype=torch.float64),
        predicted_temperature=torch.tensor([1001.0], dtype=torch.float64),
        target_temperature=torch.tensor([1000.8], dtype=torch.float64),
        pressure=torch.tensor([101325.0], dtype=torch.float64),
        dt=torch.tensor([1.0e-6], dtype=torch.float64),
        molecular_weights=torch.tensor([2.0, 28.0], dtype=torch.float64),
        zero_species_enthalpy=torch.zeros(2, dtype=torch.float64),
    )

    source = report["terms"]["source"]
    torch.testing.assert_close(source["raw_mae"], torch.tensor(20.0, dtype=torch.float64))
    torch.testing.assert_close(source["scaled"], torch.tensor(200.0, dtype=torch.float64))
    torch.testing.assert_close(source["contribution"], torch.tensor(10.0, dtype=torch.float64))
    assert report["terms"]["mixture_molecular_weight"]["scale"].item() == 1.0e-3
    assert report["terms"]["density_increment"]["scale"].item() == 1.0e-3
    report["total"].backward()
    assert predicted_delta_y.grad is not None
    assert torch.all(torch.isfinite(predicted_delta_y.grad))


def test_density_loss_remains_defined_for_untrained_negative_temperature_head():
    config = training.PositiveIntervalTrainingConfig(
        density_increment_loss_weight=0.02,
        density_increment_loss_scale=1.0e-3,
    )
    predicted_temperature = torch.tensor([-500.0], dtype=torch.float64, requires_grad=True)
    report = training._deployment_loss_for_batch(
        loss_config=training._deployment_loss_config(config),
        current_y=torch.tensor([[0.7, 0.3]], dtype=torch.float64),
        predicted_delta_y=torch.tensor([[-1.0e-4, 1.0e-4]], dtype=torch.float64),
        target_delta_y=torch.tensor([[-0.8e-4, 0.8e-4]], dtype=torch.float64),
        current_temperature=torch.tensor([1000.0], dtype=torch.float64),
        predicted_temperature=predicted_temperature,
        target_temperature=torch.tensor([1000.8], dtype=torch.float64),
        pressure=torch.tensor([101325.0], dtype=torch.float64),
        dt=torch.tensor([1.0e-6], dtype=torch.float64),
        molecular_weights=torch.tensor([2.0, 28.0], dtype=torch.float64),
        zero_species_enthalpy=torch.zeros(2, dtype=torch.float64),
    )
    assert torch.isfinite(report["total"])
    report["total"].backward()
    assert predicted_temperature.grad is not None
    assert torch.all(torch.isfinite(predicted_temperature.grad))


def test_deployment_temperature_matches_exported_delta_temperature_head():
    current_temperature = torch.tensor([900.0, 1200.0], dtype=torch.float64)
    output = {
        "delta_temperature": torch.tensor([[2.5], [-1.5]], dtype=torch.float64)
    }
    result = training._deployment_endpoint_temperature(
        output,
        prediction=torch.tensor([[9.0, 0.0], [8.0, 0.0]], dtype=torch.float64),
        current_temperature=current_temperature,
        state_mean=torch.tensor([1000.0, 101325.0], dtype=torch.float64),
        state_std=torch.tensor([100.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result,
        torch.tensor([902.5, 1198.5], dtype=torch.float64),
    )


def test_flattened_report_exposes_raw_scaled_weighted_scale_and_weight():
    config = training.PositiveIntervalTrainingConfig(
        source_loss_weight=0.05,
        source_loss_scale=0.1,
    )
    loss_config = training._deployment_loss_config(config)
    accumulator = training._empty_loss_accumulator()
    accumulator["source"] = {
        "raw_mae": 4.0,
        "scaled": 40.0,
        "weighted": 2.0,
    }
    flattened = training._flatten_loss_report("validation", accumulator, 2, loss_config)
    assert flattened["validation_source_raw_mae"] == 2.0
    assert flattened["validation_source_scaled"] == 20.0
    assert flattened["validation_source_weighted"] == 1.0
    assert flattened["validation_source_scale"] == 0.1
    assert flattened["validation_source_weight"] == 0.05
