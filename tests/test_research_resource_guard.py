"""Deterministic worker-exit race, with no real GPU or process termination."""
import os
from unittest.mock import Mock
import pytest
pytest.importorskip("torch")
from benchmarks.offline_accuracy.paper_baseline.hidden_features import campaign


def test_worker_exit_during_gpu_poll_is_not_foreign_contention(tmp_path, monkeypatch):
    child = Mock(pid=1010, returncode=0)
    # Worker alive at loop entry, exits while the two external views are read.
    child.poll.side_effect = [None, 0, 0, 0]
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(campaign, "command", lambda *a, **k: ["unused"])
    monkeypatch.setattr(campaign, "process_identity", lambda pid: "start")
    monkeypatch.setattr(campaign.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(campaign, "processes_on_gpu", Mock(side_effect=[set(), {1010}]))
    monkeypatch.setattr(campaign, "descendants", lambda pid: {os.getpid()})
    campaign.run_worker(tmp_path, tmp_path, tmp_path / "output", [], tmp_path / "state.json", 0)
    child.terminate.assert_not_called()


def test_live_foreign_gpu_process_stops_only_our_child(tmp_path, monkeypatch):
    child = Mock(pid=1010, returncode=None)
    child.poll.return_value = None
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(campaign, "command", lambda *a, **k: ["unused"])
    monkeypatch.setattr(campaign, "process_identity", lambda pid: "start")
    monkeypatch.setattr(campaign.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(campaign, "processes_on_gpu", Mock(side_effect=[set(), {9999}]))
    monkeypatch.setattr(campaign, "descendants", lambda pid: {os.getpid(),1010})
    with pytest.raises(RuntimeError, match="Resource contention"):
        campaign.run_worker(tmp_path, tmp_path, tmp_path / "output", [], tmp_path / "state.json", 0)
    child.terminate.assert_called_once()
