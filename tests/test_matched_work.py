import numpy as np
import pytest

from benchmarks.offline_accuracy.paper_baseline.matched_work.plan import configuration, update_policy, batch_indices


def test_update_schedule_boundaries():
    state = configuration("fuel-state", 20261011)
    power = configuration("fuel-power", 20261011)
    assert update_policy(state, 0) == (1e-3, True)
    assert update_policy(state, 1999) == (1e-3, False)
    assert update_policy(state, 2000) == (1e-4, True)
    assert update_policy(state, 2001) == (1e-4, False)
    assert update_policy(state, 4000)[1]
    assert update_policy(power, 1499) == (1e-3, False)
    assert update_policy(power, 1500) == (1e-4, False)
    np.testing.assert_allclose(update_policy(power, 4500)[0], 1e-6, rtol=1e-15)
    with pytest.raises(ValueError):
        update_policy(state, 6000)


def test_equal_presentations_and_complete_pool_passes():
    config = dict(configuration("fuel-state", 20261011), batch_size=10, updates=60)
    for size in (50, 200):
        batches = list(batch_indices(size, config))
        assert len(batches) == 60 and sum(map(len, batches)) == 600
        for start in range(0, len(batches), size//10):
            np.testing.assert_array_equal(np.sort(np.concatenate(batches[start:start+size//10])), np.arange(size))
        np.testing.assert_array_equal(batches, list(batch_indices(size, config)))
    with pytest.raises(ValueError):
        list(batch_indices(51, config))


def test_gpu_fit_uses_frozen_normalization_and_replays(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required; no CPU training fallback")
    from benchmarks.offline_accuracy.paper_baseline.matched_work.fit import fit, preprocessing_hash
    from benchmarks.offline_accuracy.paper_baseline.fit import reload_model, prediction
    from benchmarks.offline_accuracy.paper_baseline.coordinates import preprocessing
    states = np.array([[900.+i*50, 100000.+i*100, .1+i*.01, .9-i*.01] for i in range(8)])
    full = dict(states=states, delta=np.tile([1e-6, 0.], (8, 1)), source_indices=np.arange(8))
    small = {key:value[:4] for key,value in full.items()}
    config = dict(configuration("fuel-state", 20261011), batch_size=2, updates=6,
                  widths=[8, 8], normalization_rows=4, diagnostics_every_updates=3)
    expected = preprocessing(small, ["H2", "AR"], config)
    results = []
    for name, training in (("small", small), ("large", full)):
        destination = tmp_path / name
        destination.mkdir()
        model, prep, result = fit(training, small, small, ["H2", "AR"], config, destination)
        results.append(result)
        assert result["updates_completed"] == 6 and result["row_presentations"] == 12
        assert result["preprocessing_array_sha256"] == preprocessing_hash(expected)
        restored, saved = reload_model(destination, config)
        np.testing.assert_array_equal(prediction(model, prep, states, config)[0], prediction(restored, saved, states, config)[0])
    assert results[0]["initial_weights_sha256"] == results[1]["initial_weights_sha256"]


def test_campaign_is_eight_fixed_fits(tmp_path):
    pytest.importorskip("cantera")
    pytest.importorskip("torch")
    from benchmarks.offline_accuracy.paper_baseline.matched_work.campaign import commands
    stages = commands(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "new")
    assert len(stages) == 8 and len({name for name, _ in stages}) == 8
    assert not (tmp_path / "new").exists()
    assert all(command[-1] == "--execute" for _, command in stages)
