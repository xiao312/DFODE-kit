from copy import deepcopy
import json
import pytest
from benchmarks.offline_accuracy.paper_baseline.matched_targets.checks import matrix_names
from benchmarks.offline_accuracy.paper_baseline.matched_targets.review import collect


def test_incomplete_matrix_is_rejected(tmp_path):
    (tmp_path/'campaign-status.json').write_text(json.dumps(dict(status='complete',stages=[])))
    (tmp_path/'verification.json').write_text(json.dumps(dict(status='verified',results=[])))
    with pytest.raises(ValueError,match='24-fit'):
        collect(tmp_path)


def test_complete_matrix_names_are_unique_and_ordered():
    names=matrix_names()
    assert len(names)==len(set(names))==24
    assert names[0]=='20261011-state-boxcox-coordinate'
    assert names[-1]=='20261012-gbct-state'
