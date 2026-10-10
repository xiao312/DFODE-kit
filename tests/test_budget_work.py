from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.offline_accuracy.paper_baseline.matched_work.plan import configuration, extended_configuration, update_policy, batch_indices
from benchmarks.offline_accuracy.paper_baseline.matched_work.budget_checks import check_baseline, check_pair


@pytest.mark.parametrize("recipe", ["fuel-state", "fuel-power"])
def test_budget_changes_only_work_and_experiment_identity(recipe):
    small = configuration(recipe, 20261011)
    large = extended_configuration(recipe, 20261011)
    assert {key for key in small if small[key] != large[key]} == {"experiment", "updates"}
    assert large["updates"]*large["batch_size"] == 180000000
    for completed in (0, 1, 1499, 1500, 1999, 2000, 3999, 4000, 4499, 4500, 5999):
        assert update_policy(small, completed) == update_policy(large, completed*3)
    a = islice(batch_indices(200000, small), 22)
    b = islice(batch_indices(200000, large), 22)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_rejects_a_different_baseline():
    result = dict(config=configuration("fuel-state", 20261011), training_count=200000,
        development_count=1023, independent_test_count=0, updates_completed=6000,
        row_presentations=60000000, hashes={"dataset":"same"})
    check_baseline(result, "fuel-state", 20261011, {"dataset":"same"})
    for changes in ({"training_count":50000}, {"row_presentations":1}, {"independent_test_count":1}):
        with pytest.raises(ValueError):
            check_baseline(dict(result, **changes), "fuel-state", 20261011, {"dataset":"same"})


def test_four_budget_commands_are_read_only(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("cantera")
    from benchmarks.offline_accuracy.paper_baseline.matched_work.budget_campaign import commands
    args = SimpleNamespace(**{key:tmp_path / key for key in ("campaign", "previous", "original", "base_root", "baseline", "output")})
    rows = commands(args)
    assert len(rows) == 4 and len({name for name, _ in rows}) == 4
    assert all("--budget-baseline" in command for _, command in rows)
    assert not args.output.exists()


def test_pair_checks_frozen_arrays_and_weights(tmp_path):
    old_path, new_path = tmp_path / 'old', tmp_path / 'new'
    for directory in (old_path, new_path):
        directory.mkdir()
        np.save(directory / 'training-indices.npy', np.arange(5))
        np.save(directory / 'normalization-indices.npy', np.arange(3))
        np.savez(directory / 'development-predictions.npz', source_indices=np.arange(8, 10))
    common = dict(hashes={'data':'checked'}, initial_weights_sha256='same-weights',
                  preprocessing_array_sha256='same-scalers', training_count=200000,
                  development_count=1023, independent_test_count=0, effective_batch_size=10000)
    old = dict(common, config=configuration('fuel-state',20261011))
    new = dict(common, config=extended_configuration('fuel-state',20261011),
               updates_completed=18000, row_presentations=180000000)
    check_pair(old_path,new_path,old,new)
    for key in ('initial_weights_sha256','preprocessing_array_sha256'):
        with pytest.raises(ValueError):
            check_pair(old_path,new_path,old,dict(new,**{key:'changed'}))
    np.save(new_path / 'training-indices.npy', np.arange(5)+1)
    with pytest.raises(AssertionError):
        check_pair(old_path,new_path,old,new)
