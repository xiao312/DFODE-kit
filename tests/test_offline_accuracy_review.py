import copy
import pytest
from benchmarks.offline_accuracy.review import extend_snapshot, compile_rows


def test_review_preserves_identity_prior_queries_and_nulls():
    original = {"id": "stable", "title": "Existing title", "report": {"asOf": "2026-10-08"},
                "queries": {"prior": {"rows": [{"unchanged": 1}]}}}
    result = extend_snapshot(copy.deepcopy(original), {"offline_models": [{"rate": None}]}, [])
    assert result["id"] == original["id"] and result["title"] == original["title"]
    assert result["queries"]["prior"] == original["queries"]["prior"]
    assert result["queries"]["offline_models"]["rows"][0]["rate"] is None
    assert result["buildStatus"] == "updating"


def test_review_rejects_missing_seeds():
    with pytest.raises(ValueError, match="both predeclared seeds"):
        compile_rows([])
