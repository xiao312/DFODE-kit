import json
from pathlib import Path
import time

import numpy as np
import pytest

ct = pytest.importorskip("cantera")
from benchmarks.flame_conditioning.chemistry import EndpointIntegrator
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.parallel_labels import run
from benchmarks.flame_conditioning.parallel_labels.storage import (
    assert_prefix, commit_chunk, coverage, missing_ranges, read_chunk, same_dataset_config)
from benchmarks.flame_conditioning.parallel_labels.worker import chunks, label_rows


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    gas = ct.Solution("h2o2.yaml")
    gas.TPX = 1100., ct.one_atm, "H2:2,O2:1,N2:3.76"
    gas.write_yaml(directory / "mechanism.yaml")
    rows = np.array([[temperature, ct.one_atm, *gas.Y] for temperature in (900., 1100., 1500.)])
    np.savez_compressed(directory / "source-states.npz", states=np.tile(rows, (2, 1)),
                        snapshot=np.repeat(["a", "b"], 3), coordinate=np.tile(np.arange(3), 2))
    manifest = dict(mechanism_sha256=sha256(directory / "mechanism.yaml"),
                    states_sha256=sha256(directory / "source-states.npz"), species_names=gas.species_names)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    config = dict(schema_version=1, seed=123, train_snapshots=["a"], validation_snapshots=["b"],
                  train_count=12, validation_count=6, interval_s=1e-6, cvode_rtol=1e-12,
                  cvode_atol=1e-21, minimum_temperature_K=290., wall_seconds=60,
                  temperature_perturbation_K=10., species_exponent_perturbation=.15,
                  nitrogen_range_padding=.1, pressure_bounds_Pa=[96000., 106000.])
    return directory, config, rows


def test_process_labels_equal_serial_and_keep_failed_rows(source):
    directory, config, rows = source
    rows = np.concatenate([rows, rows[:1]])
    rows[-1, 2] = -1  # Must be excluded, never relabeled as zero.
    expected, mask, records = label_rows(EndpointIntegrator(directory / "mechanism.yaml"), rows, 1e-6)
    predicted, accepted = np.full_like(expected, np.nan), np.zeros(len(rows), dtype=bool)
    for start, delta, flags, _ in chunks(directory / "mechanism.yaml", config, rows,
                                       [(0, 2), (2, 4)], 2, time.monotonic()+60):
        predicted[start:start+len(delta)] = delta
        accepted[start:start+len(delta)] = flags
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(accepted, mask)
    assert "error" in records[-1] and np.isnan(predicted[-1]).all()


def test_chunk_hash_and_range_guards(tmp_path):
    entry = commit_chunk(tmp_path, 2, np.ones((2, 3)), np.array([True, True]), [{"row": 2}, {"row": 3}])
    read_chunk(tmp_path, entry, 3)
    assert missing_ranges(coverage(7, [entry]), 2) == [(0, 2), (4, 6), (6, 7)]
    with pytest.raises(ValueError, match="Overlapping"):
        coverage(7, [entry, entry])
    with pytest.raises(ValueError, match="overwriting"):
        commit_chunk(tmp_path, 2, np.ones((2, 3)), np.array([True, True]), [])
    (tmp_path / entry["file"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        read_chunk(tmp_path, entry, 3)


def test_reuse_contract_and_prefix(source):
    _, config, rows = source
    same_dataset_config(dict(config, train_count=24), config)
    with pytest.raises(ValueError, match="chemistry"):
        same_dataset_config(dict(config, cvode_atol=1e-18), config)
    assert assert_prefix({"states": rows}, {"states": rows[:2]}) == 2
    with pytest.raises(ValueError, match="prefix"):
        assert_prefix({"states": rows+1}, {"states": rows[:2]})


def test_dry_run_no_writes_and_changed_resume_rejected(source, tmp_path):
    directory, config, _ = source
    output = tmp_path / "preview"
    run.preflight(directory, config, output)
    assert not output.exists()


def test_interrupted_resume_and_reuse_preserve_labels(source, tmp_path, monkeypatch):
    directory, config, _ = source
    complete = tmp_path / "complete"
    run.execute(directory, config, complete, 1, 3)
    original_chunks = run.chunks

    def interrupt(*args, **kwargs):
        iterator = original_chunks(*args, **kwargs)
        try:
            yield next(iterator)
        finally:
            iterator.close()
        raise TimeoutError("Test interruption")

    partial = tmp_path / "partial"
    monkeypatch.setattr(run, "chunks", interrupt)
    with pytest.raises(TimeoutError):
        run.execute(directory, config, partial, 1, 3)
    partial_hash = sha256(partial / "manifest.json")
    with pytest.raises(ValueError, match="identical"):
        run.preflight(directory, dict(config, cvode_rtol=1e-10), tmp_path / "bad", partial)
    monkeypatch.setattr(run, "chunks", original_chunks)
    resumed = tmp_path / "resumed"
    result = run.execute(directory, config, resumed, 2, 2, resume=partial)
    assert result["resume_manifest_sha256"] == partial_hash
    assert sha256(partial / "manifest.json") == partial_hash
    for split in ("train", "validation"):
        with np.load(complete / split / "labels.npz") as first, np.load(resumed / split / "labels.npz") as second:
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
    expanded = tmp_path / "expanded"
    result = run.execute(directory, dict(config, train_count=24), expanded, 2, 3, reuse=complete)
    assert result["splits"]["train"]["reused_rows"] == 12
    assert result["splits"]["validation"]["reused_rows"] == 6
    with np.load(complete / "train/labels.npz") as first, np.load(expanded / "train/labels.npz") as second:
        np.testing.assert_array_equal(first["delta"], second["delta"][:12])
