from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dfode_kit.evaluation import positive_interval as evaluation
from dfode_kit.models.positive_interval import NeuralPatankarIntervalModel


REACTANTS = np.asarray(
    [
        [1.0],
        [1.0],
        [0.0],
    ],
    dtype=np.float64,
)
PRODUCTS = np.asarray(
    [
        [0.0],
        [0.0],
        [2.0],
    ],
    dtype=np.float64,
)
REVERSIBLE = np.asarray([False])


def _checkpoint(**extra):
    model = NeuralPatankarIntervalModel(
        5,
        REACTANTS,
        PRODUCTS,
        REVERSIBLE,
        latent_dim=4,
        hidden_dim=8,
        extent_scale=1.0,
    )
    checkpoint = {
        "positive_model_type": "neural-patankar",
        "training_config": {
            "latent_dim": 4,
            "hidden_dim": 8,
            "extent_scale": 1.0,
            "positivity_floor": 1e-30,
            "temperature_delta_scale": 10.0,
        },
        "net": model.state_dict(),
    }
    checkpoint.update(extra)
    return checkpoint


def _load(monkeypatch, checkpoint):
    stoich = SimpleNamespace(
        reactants=REACTANTS,
        products=PRODUCTS,
        reversible=REVERSIBLE,
    )
    monkeypatch.setattr(evaluation, "reaction_stoichiometry", lambda gas: stoich)
    gas = SimpleNamespace(molecular_weights=np.ones(3, dtype=np.float64))
    return evaluation._load_positive_model(
        checkpoint,
        gas,
        input_dim=5,
        device=torch.device("cpu"),
    )


def test_evaluation_loader_restores_top_level_smooth_safe_settings(monkeypatch):
    model = _load(
        monkeypatch,
        _checkpoint(availability_mode="smooth-safe", availability_p_norm=20.0),
    )

    assert model.availability_mode == "smooth-safe"
    assert model.availability_p_norm == 20.0


def test_evaluation_loader_accepts_training_config_settings(monkeypatch):
    checkpoint = _checkpoint()
    checkpoint["training_config"].update(
        availability_mode="smooth-safe",
        availability_p_norm=24.0,
    )

    model = _load(monkeypatch, checkpoint)

    assert model.availability_mode == "smooth-safe"
    assert model.availability_p_norm == 24.0


def test_evaluation_loader_keeps_legacy_checkpoint_defaults(monkeypatch):
    model = _load(monkeypatch, _checkpoint())

    assert model.availability_mode == "hard"
    assert model.availability_p_norm == 32.0


@pytest.mark.parametrize(
    ("mode", "p_norm"),
    [
        ("unsupported", 32.0),
        ("smooth-safe", 0.5),
        ("smooth-safe", float("nan")),
    ],
)
def test_evaluation_loader_rejects_invalid_availability_settings(
    monkeypatch,
    mode,
    p_norm,
):
    with pytest.raises(ValueError, match="availability_"):
        _load(
            monkeypatch,
            _checkpoint(availability_mode=mode, availability_p_norm=p_norm),
        )
