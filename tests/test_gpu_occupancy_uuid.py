"""Telemetry must accept the same UUID selector as the isolated CUDA runtime."""
import os
import pytest
pytest.importorskip('torch')
from benchmarks.offline_accuracy.paper_baseline.decoder_loss import fit


@pytest.mark.parametrize('selector',['2','GPU-6d8d1985-1dfa-e1ba-1b73-849786ca2dba'])
def test_occupancy_accepts_index_or_uuid(monkeypatch,selector):
    uuid='GPU-6d8d1985-1dfa-e1ba-1b73-849786ca2dba'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',selector)
    monkeypatch.setattr(fit,'inventory',lambda:[dict(index='2',uuid=uuid)])
    monkeypatch.setattr(fit.subprocess,'check_output',lambda *a,**k:f'{os.getpid()}, {uuid}\n9999999, GPU-other\n')
    result=fit.occupancy_sample(1000)
    assert result['gpu']['uuid']==uuid and result['other_process_count']==0


def test_occupancy_unknown_selector_is_explicit(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','missing')
    monkeypatch.setattr(fit,'inventory',lambda:[dict(index='2',uuid='GPU-known')])
    with pytest.raises(ValueError,match='selected GPU'):fit.occupancy_sample(1)
