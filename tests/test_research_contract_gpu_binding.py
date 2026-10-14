"""Command-level regression for one physical GPU behind isolated enumeration."""
from pathlib import Path
import pytest
from benchmarks.research_contract import sandbox

UUID = 'GPU-6d8d1985-1dfa-e1ba-1b73-849786ca2dba'


def build(tmp_path, monkeypatch, response=UUID+'\n', gpu=2):
    source=tmp_path/'project';source.mkdir()
    inputs=tmp_path/'input';inputs.mkdir()
    output=tmp_path/'output';output.mkdir()
    monkeypatch.setattr(sandbox.subprocess,'check_output',lambda *a,**k:response)
    monkeypatch.setattr(Path,'exists',lambda self:True)
    return sandbox.command(source,inputs,output,['-c','pass'],gpu=gpu)


def test_nonzero_gpu_uses_uuid_and_keeps_one_device_mount(tmp_path,monkeypatch):
    command=build(tmp_path,monkeypatch)
    start=command.index('CUDA_VISIBLE_DEVICES')
    assert command[start+1]==UUID
    mounts=[command[i+1] for i,v in enumerate(command) if v=='--dev-bind']
    assert mounts==['/dev/nvidia2','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
    assert '--unshare-all' in command and '--die-with-parent' in command
    assert '/code/benchmarks' in command and '/code/dfode_kit' in command
    assert command.count('--bind')==1  # only candidate output is writable


@pytest.mark.parametrize('reply',['','2','GPU-not-a-uuid',UUID+'\n'+UUID])
def test_bad_or_ambiguous_uuid_fails_closed(tmp_path,monkeypatch,reply):
    with pytest.raises(ValueError):build(tmp_path,monkeypatch,reply)


def test_negative_gpu_fails_closed(tmp_path,monkeypatch):
    with pytest.raises(ValueError):build(tmp_path,monkeypatch,gpu=-1)
