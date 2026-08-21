from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def test_split_fluent_jvp_is_disjoint_deterministic_and_complete(tmp_path: Path):
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as handle:
        handle.attrs["schema_version"] = "fluent-di-jvp-v1"
        handle.attrs["n_pairs"] = 6
        pairs = handle.create_group("pairs")
        pairs.create_dataset("x", data=np.arange(12, dtype=np.float64).reshape(6, 2))
        pairs.create_dataset("dt", data=np.full(6, 1.0e-6))
        metadata = handle.create_group("sample_metadata")
        metadata.create_dataset("pair_id", data=np.arange(6, dtype=np.int64))
        handle.create_dataset(
            "species_names",
            data=np.asarray(["H2", "O2"], dtype=h5py.string_dtype("utf-8")),
        )

    script = Path(__file__).parents[1] / "scripts" / "split_fluent_jvp.py"
    outputs = []
    for directory_name in ("first", "second"):
        output = tmp_path / directory_name
        subprocess.run(
            [
                sys.executable,
                str(script),
                "--source",
                str(source),
                "--output-dir",
                str(output),
                "--seed",
                "7",
                "--test-fraction",
                "0.3333333333333333",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(output)

    with h5py.File(outputs[0] / "trainval.h5", "r") as trainval, h5py.File(
        outputs[0] / "test.h5", "r"
    ) as test:
        trainval_ids = np.asarray(trainval["sample_metadata/pair_id"])
        test_ids = np.asarray(test["sample_metadata/pair_id"])
        assert trainval.attrs["split_role"] == "train-validation-pool"
        assert test.attrs["split_role"] == "untouched-test"
        assert trainval["pairs/x"].shape[0] == 4
        assert test["pairs/x"].shape[0] == 2
        assert np.intersect1d(trainval_ids, test_ids).size == 0
        assert np.array_equal(np.sort(np.concatenate([trainval_ids, test_ids])), np.arange(6))
        assert list(trainval["species_names"].asstr()[:]) == ["H2", "O2"]

    with h5py.File(outputs[1] / "test.h5", "r") as repeated:
        assert np.array_equal(test_ids, repeated["sample_metadata/pair_id"][:])
    summary = json.loads((outputs[0] / "split-summary.json").read_text())
    assert summary["trainval_pairs"] == 4
    assert summary["test_pairs"] == 2
