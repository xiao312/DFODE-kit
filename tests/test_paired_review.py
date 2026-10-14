import copy
import pytest
from benchmarks.offline_accuracy.paired.review import extend, collect


def test_preserve_old_evidence_and_identity():
    original = dict(id="same-report", title="same-title", queries={
        "offline_contract": dict(rows=[dict(datasetHash="data", auditHash="audit")]),
        "old": dict(rows=[dict(value=3)])})
    updated = extend(copy.deepcopy(original), {"paired_models": [dict(testRate=None)]}, [], ("data", "audit"))
    assert updated["id"] == original["id"]
    assert updated["queries"]["old"] == original["queries"]["old"]
    assert updated["queries"]["paired_models"]["rows"][0]["testRate"] is None
    assert updated["buildStatus"] == "updating"
    with pytest.raises(ValueError, match="data differ"):
        extend(copy.deepcopy(original), {}, [], ("wrong", "audit"))


def test_partial_campaign_is_not_publishable(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect(tmp_path)
