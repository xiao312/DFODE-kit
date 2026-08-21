from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from dfode_kit.cli.commands.train_positive_interval import add_command_parser
from dfode_kit.data.fluent_sensitivity import (
    FLUENT_NATIVE_DI_BACKEND,
    FluentDIJVPBatch,
    deterministic_jvp_split_indices,
    deterministic_paired_batch_indices,
    write_fluent_di_jvp_dataset,
)
from dfode_kit.training.fluent_deployment_losses import (
    FluentJVPConsistencyConfig,
    LossTerm,
    fluent_di_jvp_consistency_losses,
)
from dfode_kit.training import positive_interval as training


def _cli_args():
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


def _jvp_batch(*, backend=FLUENT_NATIVE_DI_BACKEND, dt=1.0e-6, n=10):
    x = np.tile(np.array([1000.0, 101325.0, 0.8, 0.2]), (n, 1))
    direction = np.tile(np.array([5.0, 10.0, -0.1, 0.1]), (n, 1))
    epsilon = np.full(n, 1.0e-3)
    perturbed = x + epsilon[:, None] * direction
    endpoint = x.copy()
    endpoint[:, 2:] += np.array([-1.0e-4, 1.0e-4])
    endpoint_perturbed = perturbed.copy()
    endpoint_perturbed[:, 2:] += np.array([-1.1e-4, 1.1e-4])
    return FluentDIJVPBatch(
        x=x,
        x_perturbed=perturbed,
        phi_di_x=endpoint,
        phi_di_x_perturbed=endpoint_perturbed,
        epsilon=epsilon,
        direction=direction,
        dt=np.full(n, dt),
        species_names=("A", "B"),
        provenance={
            "label_backend": backend,
            "exact_dt_seconds": dt,
            "case_id": "test",
        },
    )


def test_identity_endpoint_map_has_zero_source_jvp_including_minus_identity():
    state = torch.tensor(
        [[1000.0, 101325.0, 0.8, 0.2]], dtype=torch.float64
    )
    epsilon = torch.tensor([1.0e-3], dtype=torch.float64)
    direction = torch.tensor(
        [[5.0, 10.0, -0.1, 0.1]], dtype=torch.float64
    )
    perturbed = state + epsilon[:, None] * direction
    report = fluent_di_jvp_consistency_losses(
        current_state=state,
        perturbed_state=perturbed,
        predicted_endpoint=state,
        predicted_perturbed_endpoint=perturbed,
        target_di_endpoint=state,
        target_di_perturbed_endpoint=perturbed,
        epsilon=epsilon,
        dt=torch.tensor([1.0e-6], dtype=torch.float64),
        config=FluentJVPConsistencyConfig(
            endpoint=LossTerm(0.0, 1.0),
            source=LossTerm(1.0, 1.0),
        ),
        label_backend=FLUENT_NATIVE_DI_BACKEND,
    )
    torch.testing.assert_close(
        report["derived"]["predicted_source_jvp"],
        torch.zeros((1, 2), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert report["total"].item() == 0.0


def test_deterministic_pair_split_and_ddp_batches_do_not_resample():
    train_a, validation_a = deterministic_jvp_split_indices(23, seed=17)
    train_b, validation_b = deterministic_jvp_split_indices(23, seed=17)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(validation_a, validation_b)
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(23))

    per_rank = []
    for rank in range(4):
        batches = deterministic_paired_batch_indices(
            train_a,
            batch_size=3,
            seed=31,
            epoch=4,
            rank=rank,
            world_size=4,
        )
        flattened = np.concatenate(batches)
        assert np.unique(flattened).size == flattened.size
        per_rank.append(set(flattened.tolist()))
    assert all(
        per_rank[left].isdisjoint(per_rank[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_jvp_training_requires_native_main_labels_and_matching_contract(tmp_path):
    path = tmp_path / "pairs.h5"
    write_fluent_di_jvp_dataset(path, _jvp_batch())
    config = training.PositiveIntervalTrainingConfig(
        jvp_dataset=str(path), source_jvp_loss_weight=0.1
    )
    native = {"label_backend": "fluent-native-direct-integration"}
    with pytest.raises(ValueError, match="Fluent native direct integration"):
        training._require_deployment_label_sources(
            config, {"label_backend": "cantera-cvode"}, native
        )

    prepared = training._prepare_jvp_training_data(
        config,
        species_names=("A", "B"),
        train_dt=np.full(4, 1.0e-6),
        validation_dt=np.full(2, 1.0e-6),
    )
    assert prepared is not None
    assert prepared.train_indices.size == 8
    assert prepared.validation_indices.size == 2
    prepared_alias = training._prepare_jvp_training_data(
        config,
        species_names=("a", "b"),
        train_dt=np.full(4, 1.0e-6),
        validation_dt=np.full(2, 1.0e-6),
    )
    assert prepared_alias is not None
    with pytest.raises(ValueError, match="species order"):
        training._prepare_jvp_training_data(
            config,
            species_names=("B", "A"),
            train_dt=np.full(4, 1.0e-6),
            validation_dt=np.full(2, 1.0e-6),
        )
    with pytest.raises(ValueError, match="exactly match JVP"):
        training._prepare_jvp_training_data(
            config,
            species_names=("A", "B"),
            train_dt=np.full(4, 2.0e-6),
            validation_dt=np.full(2, 1.0e-6),
        )


def test_jvp_cli_and_disabled_legacy_path_are_backward_compatible(monkeypatch):
    parser = argparse.ArgumentParser()
    add_command_parser(parser.add_subparsers(dest="command"))
    defaults = parser.parse_args(_cli_args())
    assert defaults.jvp_dataset is None
    assert defaults.source_jvp_loss_weight == 0.0
    assert defaults.source_jvp_loss_scale == 1.0

    configured = parser.parse_args(
        _cli_args()
        + [
            "--jvp-dataset",
            "pairs.h5",
            "--source-jvp-loss-weight",
            "0.25",
            "--source-jvp-loss-scale",
            "1000",
        ]
    )
    assert configured.jvp_dataset == "pairs.h5"
    assert configured.source_jvp_loss_weight == 0.25
    assert configured.source_jvp_loss_scale == 1000.0

    monkeypatch.setattr(
        training,
        "load_fluent_di_jvp_dataset",
        lambda _path: pytest.fail("legacy path must not load JVP data"),
    )
    assert training._prepare_jvp_training_data(
        training.PositiveIntervalTrainingConfig(),
        species_names=("A", "B"),
        train_dt=np.array([1.0e-6]),
        validation_dt=np.array([1.0e-6]),
    ) is None
    with pytest.raises(ValueError, match="jvp-dataset"):
        training._validate_jvp_configuration(
            training.PositiveIntervalTrainingConfig(source_jvp_loss_weight=0.1)
        )
