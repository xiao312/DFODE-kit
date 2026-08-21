from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import torch

from dfode_kit.cli.commands.train_positive_interval import add_command_parser
from dfode_kit.training import positive_interval as training
from scripts.export_fluent_artifact import (
    FluentNeuralPatankarWrapper,
    _availability_settings,
)


def test_training_cli_preserves_hard_default_and_accepts_smooth_safe():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_command_parser(subparsers)
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
    hard = parser.parse_args(required)
    smooth = parser.parse_args(
        required
        + [
            "--availability-mode",
            "smooth-safe",
            "--availability-p-norm",
            "24",
        ]
    )
    assert hard.availability_mode == "hard"
    assert hard.availability_p_norm == 32.0
    assert smooth.availability_mode == "smooth-safe"
    assert smooth.availability_p_norm == 24.0


def test_training_build_passes_smooth_safe_configuration(monkeypatch):
    stoich = SimpleNamespace(
        reactants=np.asarray([[1.0], [1.0], [0.0]]),
        products=np.asarray([[0.0], [0.0], [2.0]]),
        reversible=np.asarray([0.0]),
        net=np.asarray([[-1.0], [-1.0], [1.0]]),
    )
    monkeypatch.setattr(training, "reaction_stoichiometry", lambda gas: stoich)
    gas = SimpleNamespace(molecular_weights=np.ones(3))
    config = training.PositiveIntervalTrainingConfig(
        availability_mode="smooth-safe",
        availability_p_norm=24.0,
    )
    model = training._build_model(config, gas, input_dim=5)
    assert model.availability_mode == "smooth-safe"
    assert model.availability_p_norm == 24.0


def test_checkpoint_availability_settings_are_backward_compatible():
    assert _availability_settings({"training_config": {}}) == (
        "hard",
        32.0,
        1e-30,
    )
    assert _availability_settings(
        {
            "availability_mode": "smooth-safe",
            "availability_p_norm": 20.0,
            "training_config": {"positivity_floor": 1e-24},
        }
    ) == ("smooth-safe", 20.0, 1e-24)


def test_export_wrapper_matches_trained_smooth_safe_model():
    reactants = np.asarray([[1.0], [1.0], [0.0]], dtype=np.float64)
    products = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float64)
    reversible = np.asarray([0.0], dtype=np.float64)
    model = training.NeuralPatankarIntervalModel(
        input_dim=5,
        reactant_mass_matrix=reactants,
        product_mass_matrix=products,
        reaction_reversible=reversible,
        latent_dim=4,
        hidden_dim=8,
        extent_scale=1.0,
        demand_bias_init=0.0,
        availability_mode="smooth-safe",
        availability_p_norm=16.0,
    ).eval()
    wrapper = FluentNeuralPatankarWrapper(
        model,
        np.zeros(5),
        np.ones(5),
        0.0,
        1.0,
        include_temperature=False,
        availability_mode="smooth-safe",
        availability_p_norm=16.0,
    ).eval()
    physical = torch.tensor(
        [[1000.0, 101325.0, 0.2, 0.2, 0.6, 1.0]],
        dtype=torch.float32,
    )
    species = torch.tensor([[0.2, 0.2, 0.6]], dtype=torch.float64)
    normalized = physical[:, :-1]
    log_dt = torch.zeros((1, 1), dtype=torch.float32)
    with torch.inference_mode():
        expected = model(
            normalized,
            log_dt,
            current_species=species,
        )["delta_species"]
        actual = wrapper(physical, species)
        dense = wrapper.dense_reference(physical, species)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-15)
    torch.testing.assert_close(dense, expected, rtol=0.0, atol=1e-15)
