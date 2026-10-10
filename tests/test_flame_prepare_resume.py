import json
import sys

import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning import prepare


def test_checkpoint_spacing_is_bounded_and_does_not_grow_per_label_cost():
    assert prepare.checkpoint_stride(3) == 100
    assert prepare.checkpoint_stride(10000) == 100
    assert prepare.checkpoint_stride(50000) == 250
    assert prepare.checkpoint_stride(200000) == 1000
    assert prepare.checkpoint_stride(1000000) == 1000


def test_resume_copies_stopped_run_and_preserves_completed_labels(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "mechanism.yaml").write_text("fixture only")
    np.savez(source / "source-states.npz", snapshot=np.array(["a"]))
    manifest = {"species_names": ["A", "B"],
                "mechanism_sha256": prepare.sha256(source / "mechanism.yaml"),
                "states_sha256": prepare.sha256(source / "source-states.npz")}
    (source / "manifest.json").write_text(json.dumps(manifest))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"wall_seconds": 2, "cvode_rtol": 1e-10,
                                  "cvode_atol": 1e-20, "interval_s": 1e-6}))
    clock = [0.0]
    calls = []

    class Integrator:
        def __init__(self, *args):
            pass

        def advance(self, state, interval):
            calls.append(state["T"])
            clock[0] += 1
            return {"delta": np.array([-.01, .01]), "diagnostics": {
                "minimum_mass_fraction": .4, "mass_delta_sum": 0,
                "element_delta_max": 0, "temperature_change_K": 0,
                "relative_density_change": 0}}

    def sample(source_data, names, settings, split):
        temperatures = [300, 400, 500] if split == "train" else [600, 700]
        states = np.array([[t, 101325, .5, .5] for t in temperatures])
        return states, {"snapshot": np.array(["a"] * len(states))}, {}

    monkeypatch.setattr(prepare, "EndpointIntegrator", Integrator)
    monkeypatch.setattr(prepare, "sample_split", sample)
    monkeypatch.setattr(prepare, "validate_config", lambda *args: None)
    monkeypatch.setattr(prepare, "source_revision", lambda: {"revision": "fixture"})
    monkeypatch.setattr(prepare.time, "monotonic", lambda: clock[0])
    stopped, completed = tmp_path / "stopped", tmp_path / "completed"
    command = ["prepare", str(source), "--config", str(config)]
    monkeypatch.setattr(sys, "argv", command + ["--output", str(stopped)])
    prepare.main()
    original = {p.relative_to(stopped): p.read_bytes() for p in stopped.rglob("*") if p.is_file()}
    assert json.loads((stopped / "manifest.json").read_text())["status"] == "time_limit"
    assert calls == [300, 400]
    monkeypatch.setattr(sys, "argv", command + ["--output", str(completed), "--resume-source", str(stopped)])
    prepare.main()
    assert calls == [300, 400, 500, 600, 700]
    assert original == {p.relative_to(stopped): p.read_bytes() for p in stopped.rglob("*") if p.is_file()}
    result = json.loads((completed / "manifest.json").read_text())
    assert result["status"] == "complete"
    assert result["elapsed_seconds"] == 5
    assert len(result["continuations"]) == 1
    assert result["splits"]["train"]["labels_completed"] == 3
    assert len((completed / "train" / "label-records.jsonl").read_text().splitlines()) == 3
    assert np.load(completed / "train" / "labels.npz")["accepted"].all()
