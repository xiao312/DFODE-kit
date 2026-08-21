from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import cantera as ct
import h5py
import numpy as np
import pytest
import torch

from dfode_kit.data.fluent_sensitivity import (
    FLUENT_NATIVE_DI_BACKEND,
    FluentDIJVPBatch,
    write_fluent_di_jvp_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_positive_interval_distributed.py"
LAUNCHER = ROOT / "platforms" / "scnet" / "slurm" / "run_positive_interval_single_node.slurm"
CFD_DISTRIBUTED_LAUNCHER = (
    ROOT
    / "platforms"
    / "scnet"
    / "slurm"
    / "run_cfd_perturb_distributed_train.slurm"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "train_positive_interval_distributed", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _required_args(tmp_path: Path) -> list[str]:
    return [
        "--train-source",
        str(tmp_path / "train.h5"),
        "--validation-source",
        str(tmp_path / "validation.h5"),
        "--mechanism",
        "h2o2.yaml",
        "--output",
        str(tmp_path / "best.pt"),
    ]


def _write_endpoint_dataset(path: Path, states: np.ndarray, species, phase):
    target = states.copy()
    dt = np.full(states.shape[0], 1.0e-6, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle.attrs["label_backend"] = FLUENT_NATIVE_DI_BACKEND
        handle.attrs["phase_name"] = phase
        handle.create_dataset(
            "species_names",
            data=np.asarray(species, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        handle.create_dataset("dt_bin_edges", data=np.asarray([1.0e-6, 2.0e-6]))
        pairs = handle.create_group("pairs")
        pairs.create_dataset("current_states", data=states)
        pairs.create_dataset("target_states", data=target)
        pairs.create_dataset("dt", data=dt)
        pairs.create_dataset("dt_bin", data=np.zeros(states.shape[0], dtype=np.int32))
        h_before = np.linspace(1.0e6, 1.0001e6, states.shape[0])
        pairs.create_dataset("h_total_before", data=h_before)
        pairs.create_dataset(
            "h_total_after",
            data=h_before + np.linspace(-2.0, 2.0, states.shape[0]),
        )


def _write_smoke_inputs(tmp_path: Path):
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1100.0, ct.one_atm, "H2:2,O2:1,N2:3.76"
    base = np.concatenate(([gas.T, gas.P], gas.Y))
    train = np.tile(base, (8, 1))
    validation = np.tile(base, (4, 1))
    train[:, 0] += np.arange(train.shape[0], dtype=np.float64)
    validation[:, 0] += np.arange(validation.shape[0], dtype=np.float64)
    train_path = tmp_path / "train.h5"
    validation_path = tmp_path / "validation.h5"
    _write_endpoint_dataset(train_path, train, gas.species_names, gas.name)
    _write_endpoint_dataset(
        validation_path, validation, gas.species_names, gas.name
    )

    x = np.tile(base, (5, 1))
    direction = np.zeros_like(x)
    direction[:, 0] = 1.0
    epsilon = np.full(x.shape[0], 1.0e-3)
    perturbed = x + epsilon[:, None] * direction
    jvp_path = tmp_path / "jvp.h5"
    write_fluent_di_jvp_dataset(
        jvp_path,
        FluentDIJVPBatch(
            x=x,
            x_perturbed=perturbed,
            phi_di_x=x.copy(),
            phi_di_x_perturbed=perturbed.copy(),
            epsilon=epsilon,
            direction=direction,
            dt=np.full(x.shape[0], 1.0e-6),
            species_names=gas.species_names,
            provenance={
                "label_backend": FLUENT_NATIVE_DI_BACKEND,
                "exact_dt_seconds": 1.0e-6,
                "case_id": "distributed-cpu-smoke",
            },
        ),
    )
    return train_path, validation_path, jvp_path, gas.name


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_parser_defaults_and_new_config_forwarding(tmp_path):
    module = _load_script_module()
    legacy = module.parse_args(_required_args(tmp_path))
    assert legacy.thermochemical_output_mode == "delta-temperature"
    assert legacy.total_enthalpy_delta_scale == 1.0e3
    assert legacy.total_enthalpy_delta_loss_weight == 1.0
    assert legacy.jvp_dataset is None
    assert legacy.source_jvp_loss_weight == 0.0
    assert legacy.source_jvp_loss_scale == 1.0
    assert legacy.source_loss_weight == 0.0
    assert legacy.row_tail_relative_delta_loss_weight == 0.0
    assert legacy.row_tail_relative_delta_fraction == 0.1
    assert legacy.row_tail_relative_delta_floor == 1.0e-8
    assert legacy.row_tail_relative_delta_selection_weight == 0.0
    assert legacy.mixture_molecular_weight_loss_weight == 0.0
    assert legacy.density_increment_loss_weight == 0.0
    assert legacy.normalization_source == "train"
    assert legacy.device == "cuda"

    configured = module.parse_args(
        _required_args(tmp_path)
        + [
            "--thermochemical-output-mode",
            "delta-h-total",
            "--total-enthalpy-delta-scale",
            "250",
            "--total-enthalpy-delta-loss-weight",
            "0.4",
            "--jvp-dataset",
            "pairs.h5",
            "--source-jvp-loss-weight",
            "0.2",
            "--source-jvp-loss-scale",
            "50",
            "--source-loss-weight",
            "0.3",
            "--source-loss-scale",
            "4",
            "--row-tail-relative-delta-loss-weight",
            "0.2",
            "--row-tail-relative-delta-fraction",
            "0.15",
            "--row-tail-relative-delta-floor",
            "2e-8",
            "--row-tail-relative-delta-selection-weight",
            "0.3",
            "--mixture-molecular-weight-loss-weight",
            "0.2",
            "--mixture-molecular-weight-loss-scale",
            "0.5",
            "--normalization-source",
            "initial-checkpoint",
        ]
    )
    config = module._training_config_from_args(configured, world_size=4)
    assert config.batch_size == 4 * configured.batch_size_per_device
    assert config.thermochemical_output_mode == "delta-h-total"
    assert config.total_enthalpy_delta_scale == 250.0
    assert config.total_enthalpy_delta_loss_weight == 0.4
    assert config.jvp_dataset == "pairs.h5"
    assert config.source_jvp_loss_weight == 0.2
    assert config.source_jvp_loss_scale == 50.0
    assert config.source_loss_weight == 0.3
    assert config.source_loss_scale == 4.0
    assert config.row_tail_relative_delta_loss_weight == 0.2
    assert config.row_tail_relative_delta_fraction == 0.15
    assert config.row_tail_relative_delta_floor == 2.0e-8
    assert config.row_tail_relative_delta_selection_weight == 0.3
    assert config.mixture_molecular_weight_loss_weight == 0.2
    assert config.mixture_molecular_weight_loss_scale == 0.5
    assert configured.normalization_source == "initial-checkpoint"


def test_initial_checkpoint_normalizers_are_strictly_validated(tmp_path):
    module = _load_script_module()
    args = module.parse_args(
        _required_args(tmp_path)
        + ["--normalization-source", "initial-checkpoint"]
    )
    with pytest.raises(ValueError, match="requires --initial-checkpoint"):
        module._validate_normalization_arguments(args)

    state_dimension = 5
    checkpoint = {
        "state_mean": np.arange(state_dimension, dtype=np.float64),
        "state_std": np.ones(state_dimension, dtype=np.float64),
        "log_dt_mean": np.asarray([-13.0], dtype=np.float64),
        "log_dt_std": np.asarray([0.5], dtype=np.float64),
    }
    selected = module._normalizers_from_initial_checkpoint(
        checkpoint,
        state_dimension=state_dimension,
        checkpoint_path="initial.pt",
    )
    for name, array in zip(
        ("state_mean", "state_std", "log_dt_mean", "log_dt_std"),
        selected,
    ):
        np.testing.assert_array_equal(array, checkpoint[name])
        assert array.dtype == np.float64

    missing = dict(checkpoint)
    del missing["log_dt_mean"]
    with pytest.raises(ValueError, match="missing required normalizer"):
        module._normalizers_from_initial_checkpoint(
            missing,
            state_dimension=state_dimension,
            checkpoint_path="initial.pt",
        )

    wrong_shape = dict(checkpoint, state_mean=np.zeros(state_dimension - 1))
    with pytest.raises(ValueError, match="state_mean.*shape"):
        module._normalizers_from_initial_checkpoint(
            wrong_shape,
            state_dimension=state_dimension,
            checkpoint_path="initial.pt",
        )

    nonfinite = dict(checkpoint, log_dt_mean=np.asarray([np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        module._normalizers_from_initial_checkpoint(
            nonfinite,
            state_dimension=state_dimension,
            checkpoint_path="initial.pt",
        )

    nonpositive = dict(checkpoint, state_std=np.zeros(state_dimension))
    with pytest.raises(ValueError, match="strictly positive"):
        module._normalizers_from_initial_checkpoint(
            nonpositive,
            state_dimension=state_dimension,
            checkpoint_path="initial.pt",
        )


def test_scnet_launcher_forwards_new_flags_without_changing_dcu_contract():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=dcu:4" in text
    assert '--nproc_per_node="${DCU_COUNT}"' in text
    assert "DFODE_THERMOCHEMICAL_OUTPUT_MODE:-delta-temperature" in text
    assert "DFODE_TOTAL_ENTHALPY_DELTA_SCALE:-1000.0" in text
    assert "DFODE_TOTAL_ENTHALPY_DELTA_LOSS_WEIGHT:-1.0" in text
    assert "DFODE_JVP_DATASET" in text
    assert "DFODE_SOURCE_JVP_LOSS_WEIGHT:-0.0" in text
    assert "DFODE_SOURCE_JVP_LOSS_SCALE:-1.0" in text
    assert "DFODE_SOURCE_LOSS_WEIGHT:-0.0" in text
    assert "DFODE_MIXTURE_MOLECULAR_WEIGHT_LOSS_WEIGHT:-0.0" in text
    assert "DFODE_DENSITY_INCREMENT_LOSS_WEIGHT:-0.0" in text
    assert "DFODE_ROW_TAIL_RELATIVE_DELTA_LOSS_WEIGHT:-0.0" in text
    assert "DFODE_ROW_TAIL_RELATIVE_DELTA_FRACTION:-0.1" in text
    assert "DFODE_ROW_TAIL_RELATIVE_DELTA_FLOOR:-1e-8" in text
    assert "DFODE_ROW_TAIL_RELATIVE_DELTA_SELECTION_WEIGHT:-0.0" in text


def test_cfd_distributed_launcher_forwards_normalization_source():
    text = CFD_DISTRIBUTED_LAUNCHER.read_text(encoding="utf-8")
    assert 'DFODE_NORMALIZATION_SOURCE="${DFODE_NORMALIZATION_SOURCE:-train}"' in text
    assert 'train|initial-checkpoint) ;;' in text
    assert '--normalization-source "${DFODE_NORMALIZATION_SOURCE}"' in text


def test_one_rank_cpu_total_enthalpy_and_jvp_smoke(tmp_path):
    train, validation, jvp, phase = _write_smoke_inputs(tmp_path)
    best = tmp_path / "best.pt"
    env = os.environ.copy()
    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "PYTHONPATH": str(ROOT)
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--train-source",
        str(train),
        "--validation-source",
        str(validation),
        "--mechanism",
        "h2o2.yaml",
        "--phase-name",
        phase,
        "--output",
        str(best),
        "--epochs",
        "1",
        "--batch-size-per-device",
        "2",
        "--hidden-dim",
        "8",
        "--latent-dim",
        "4",
        "--device",
        "cpu",
        "--num-threads",
        "1",
        "--thermochemical-output-mode",
        "delta-h-total",
        "--total-enthalpy-delta-scale",
        "1000",
        "--total-enthalpy-delta-loss-weight",
        "1",
        "--jvp-dataset",
        str(jvp),
        "--source-jvp-loss-weight",
        "0.01",
        "--source-jvp-loss-scale",
        "1",
        "--source-loss-weight",
        "0.1",
        "--source-loss-scale",
        "1",
        "--mixture-molecular-weight-loss-weight",
        "0.1",
        "--mixture-molecular-weight-loss-scale",
        "1",
        "--tp-loss-weight",
        "0",
        "--state-loss-weight",
        "0",
        "--delta-loss-weight",
        "0",
        "--relative-delta-loss-weight",
        "0",
        "--row-tail-relative-delta-loss-weight",
        "0.1",
        "--row-tail-relative-delta-fraction",
        "0.25",
        "--row-tail-relative-delta-floor",
        "1e-8",
        "--row-tail-relative-delta-selection-weight",
        "0.1",
        "--formation-energy-loss-weight",
        "0",
        "--direction-loss-weight",
        "0",
        "--opposing-extent-loss-weight",
        "0",
        "--log-every",
        "1",
        "--checkpoint-every",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["thermochemical_output_mode"] == (
        "delta-h-total"
    )
    assert checkpoint["training_config"]["jvp_dataset"] == str(jvp)
    assert checkpoint["training_config"][
        "row_tail_relative_delta_loss_weight"
    ] == 0.1
    record = checkpoint["history"][-1]
    assert "validation_total_enthalpy_delta_fixed_scale_loss" in record
    assert "validation_source_jvp_fixed_scale_loss" in record
    assert "train_row_tail_relative_delta_loss" in record
    assert "validation_row_relative_delta_p90" in record
    summary = json.loads(best.with_suffix(".json").read_text())
    assert summary["best_epoch"] == 1
