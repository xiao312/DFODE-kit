import hashlib
import json
from pathlib import Path
import pytest
from benchmarks.offline_accuracy.paper_baseline.reaction_attention import retry


def fixture(tmp_path,monkeypatch):
    experiment='EXP-20261014-reaction-attention'
    source=tmp_path/'research/experiments'/f'{experiment}.json'
    source.parent.mkdir(parents=True);source.write_text('{"arms":["same"]}')
    old=tmp_path/'old';old.mkdir();(old/'frozen-experiment.json').write_text(source.read_text())
    record=dict(retry_number=1,previous_output='old',output='new',scientific_plan_canonical_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (source.parent/f'{experiment}.retry1.json').write_text(json.dumps(record))
    lease=tmp_path/'runs/research-control'/f'{experiment}.lease.json';lease.parent.mkdir(parents=True)
    lease.write_text(json.dumps(dict(output=str(old),controller_pid=123,start_ticks='456')))
    monkeypatch.setattr(retry,'process_identity',lambda pid:None)
    return experiment,old,source


def test_retry_preserves_original_and_requires_new_output(tmp_path,monkeypatch):
    experiment,old,source=fixture(tmp_path,monkeypatch)
    assert retry.validate_retry(tmp_path,tmp_path/'new',experiment)['retry_number']==1
    with pytest.raises(ValueError):retry.validate_retry(tmp_path,old,experiment)
    monkeypatch.setattr(retry,'process_identity',lambda pid:'456')
    with pytest.raises(ValueError):retry.validate_retry(tmp_path,tmp_path/'new',experiment)


@pytest.mark.parametrize('artifact',['prepared','report.json','active-job.json'])
def test_retry_rejects_existing_work(tmp_path,monkeypatch,artifact):
    experiment,old,source=fixture(tmp_path,monkeypatch);(old/artifact).touch()
    with pytest.raises(ValueError):retry.validate_retry(tmp_path,tmp_path/'new',experiment)


def test_retry_rejects_changed_scientific_plan(tmp_path,monkeypatch):
    experiment,old,source=fixture(tmp_path,monkeypatch);source.write_text('{"arms":["changed"]}')
    with pytest.raises(ValueError):retry.validate_retry(tmp_path,tmp_path/'new',experiment)
