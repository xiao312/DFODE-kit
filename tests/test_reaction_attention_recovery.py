"""Reject ambiguous partial work instead of silently repeating completed fits."""
import json
import pytest
from benchmarks.offline_accuracy.paper_baseline.reaction_attention.resume import (
    check_failed_worker, check_telemetry_failure,
)


def telemetry(directory):
    for name in ('graph.npz', 'scaler.npz', 'feature-audit.npz'):
        (directory / name).touch()
    (directory / 'worker.log').write_text('occupancy_sample\nStopIteration\n')
    arm = directory / 'pool-state'
    arm.mkdir()
    (arm / 'config.json').write_text('{}')
    (arm / 'progress.jsonl').write_text(json.dumps({'update': 1000}) + '\n')
    (arm / 'gradients.jsonl').write_text(''.join(
        json.dumps({'update': step}) + '\n' for step in range(1, 1001)))
    return arm


def test_schema_recovery_requires_no_fitting_artifacts(tmp_path):
    (tmp_path / 'graph.npz').touch()
    (tmp_path / 'worker.log').write_text("KeyError: 'features'")
    check_failed_worker(tmp_path)
    (tmp_path / 'weights.pt').touch()
    with pytest.raises(ValueError):
        check_failed_worker(tmp_path)


def test_telemetry_recovery_recognizes_exact_partial_work(tmp_path):
    telemetry(tmp_path)
    check_telemetry_failure(tmp_path)


@pytest.mark.parametrize('extra', ['weights.pt', 'optimizer.pt', 'result.json'])
def test_telemetry_recovery_rejects_possible_completed_work(tmp_path, extra):
    arm = telemetry(tmp_path)
    (arm / extra).touch()
    with pytest.raises(ValueError):
        check_telemetry_failure(tmp_path)


@pytest.mark.parametrize('name', ['progress.jsonl', 'gradients.jsonl'])
def test_telemetry_recovery_rejects_changed_progress(tmp_path, name):
    arm = telemetry(tmp_path)
    (arm / name).write_text(json.dumps({'update': 999}) + '\n')
    with pytest.raises(ValueError):
        check_telemetry_failure(tmp_path)


def test_telemetry_recovery_rejects_other_failure(tmp_path):
    telemetry(tmp_path)
    (tmp_path / 'worker.log').write_text('unrelated error')
    with pytest.raises(ValueError):
        check_telemetry_failure(tmp_path)
