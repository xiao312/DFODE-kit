import numpy as np
import pytest
from benchmarks.offline_accuracy.paper_baseline.plan import configuration, epoch_policy
from benchmarks.offline_accuracy.paper_baseline.coordinates import standardize, encode, decode, preprocessing


def test_source_schedule_boundaries():
    state = configuration("fuel-state", 20261011)
    power = configuration("fuel-power", 20261011)
    assert epoch_policy(state, 500) == (1e-3, False)
    assert epoch_policy(state, 501) == (1e-4, True)
    assert epoch_policy(state, 1000) == (1e-4, False)
    assert epoch_policy(state, 1001)[1]
    np.testing.assert_allclose(epoch_policy(state, 1001)[0], 1e-5, rtol=1e-15)
    assert epoch_policy(power, 500) == (1e-4, False)
    assert epoch_policy(power, 0) == (1e-3, True)
    with pytest.raises(ValueError):
        epoch_policy(state, 1500)


def test_zero_center_is_not_rms():
    values = np.array([[1., 7.], [2., 7.], [6., 7.]])
    offset, scale, constant = standardize(values, 0, center=False)
    np.testing.assert_array_equal(offset, [0., 0.])
    np.testing.assert_allclose(scale[0], np.std(values[:, 0]))
    assert scale[1] == 1 and constant[1]
    assert scale[0] != np.sqrt(np.mean(values[:, 0]**2))


@pytest.mark.parametrize("recipe", ["fuel-state", "fuel-power"])
def test_signed_target_roundtrip(recipe):
    initial = np.array([[0., .1, 1e-30, .1]])
    delta = np.array([[1e-32, -1e-20, 1e-20, -.1]])
    decoded, _ = decode(initial, encode(initial, delta, recipe), recipe)
    np.testing.assert_allclose(decoded, delta, rtol=5e-13, atol=1e-45)


def small_training():
    return dict(states=np.array([[1000., 101325., .1, .9], [1200., 102000., .2, .8], [1300., 103000., .3, .7]]),
                delta=np.array([[1e-6, 0.], [-1e-7, 0.], [2e-6, 0.]]), source_indices=np.arange(3))


def test_output_excludes_argon_and_fits_train_scales():
    data = small_training()
    config = configuration("fuel-power", 20261011)
    prep = preprocessing(data, ["H2", "AR"], config)
    assert prep["y_offset"].shape == (1,)
    assert prep["y_offset"][0] == 0
    np.testing.assert_allclose(prep["x_scale"][1], data["states"][:, 1].std())


def test_gpu_fit_and_reload(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU required; no CPU training fallback")
    from benchmarks.offline_accuracy.paper_baseline.fit import fit, prediction, reload_model
    data = small_training()
    config = dict(configuration("fuel-power", 20261011), epochs=2, widths=[8, 8], diagnostics_every=1, batch_size=2)
    model, prep, result = fit(data, data, ["H2", "AR"], config, tmp_path)
    restored, restored_prep = reload_model(tmp_path, config)
    expected, mask = prediction(model, prep, data["states"], config)
    actual, actual_mask = prediction(restored, restored_prep, data["states"], config)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_mask, mask)
    assert result["row_presentations"] == 6 and result["updates_completed"] == 4
    assert np.all(actual[:, 1] == 0)


def test_nested_training_indices():
    pytest.importorskip("cantera")
    from benchmarks.offline_accuracy.paper_baseline.data import nested_indices
    indices = nested_indices(10010, 50100, 50000, 20261011)
    expected = np.random.default_rng(20261011).permutation(10010)[:10000]
    np.testing.assert_array_equal(indices[:10000], expected)
    assert len(indices) == len(np.unique(indices)) == 50000


def test_campaign_declares_cpu_reference_before_gpu_fits(tmp_path):
    from benchmarks.offline_accuracy.paper_baseline.campaign import commands
    stages = commands(tmp_path / "source", tmp_path / "old", tmp_path / "base", tmp_path / "new")
    assert len(stages) == 6
    assert [row[0] for row in stages[:2]] == ["prepare", "audit"]
    assert not (tmp_path / "new").exists()
    for _, timeout, command in stages[2:]:
        assert timeout == 1500
        assert command[command.index("--training-count")+1] == "50000"
        assert "--comparison-dataset" in command


def test_200k_campaign_reuses_50k_and_keeps_nested_runs(tmp_path):
    from benchmarks.offline_accuracy.paper_baseline.campaign import commands
    stages = commands(tmp_path / "source", tmp_path / "old", tmp_path / "base", tmp_path / "new",
                      200000, tmp_path / "previous", 8)
    assert stages[0][2][2] == "benchmarks.flame_conditioning.parallel_labels.run"
    assert stages[0][2][-1] == "--execute"
    assert stages[0][2][stages[0][2].index("--reuse")+1] == str(tmp_path / "previous" / "dataset")
    for _, _, command in stages[2:]:
        assert command[command.index("--training-count")+1] == "200000"
        assert "--nested-run" in command


def test_200k_selection_preserves_verified_50k_prefix(tmp_path, monkeypatch):
    pytest.importorskip("cantera")
    import json
    from benchmarks.offline_accuracy.paper_baseline import data as module
    from benchmarks.flame_conditioning.extract import sha256
    seed = 20261011
    new, old, audit, base, nested = [tmp_path / name for name in ("new", "old", "audit", "base", "nested")]
    for path in (new, old, audit, base, nested, base / "n10000-float32-state-boxcox"):
        path.mkdir(exist_ok=True)
    def write(path, value):
        path.write_text(json.dumps(value))
    source = dict(species_names=["H2", "AR"])
    config = dict(train_count=10000, wall_seconds=3600)
    old_manifest = dict(config=config, source_manifest=source)
    new_manifest = dict(config=dict(config, train_count=200500), source_manifest=source,
                        reused_dataset_manifest_sha256="verified-50k-dataset")
    write(old / "manifest.json", old_manifest)
    write(new / "manifest.json", new_manifest)
    training = dict(states=np.zeros((200000, 4)), delta=np.zeros((200000, 2)),
                    source_indices=np.arange(200000), snapshot=np.full(200000, "a"))
    development = dict(states=np.zeros((2, 4)))
    old_data = dict(train={key:values[:10000] for key, values in training.items()}, validation=development)
    new_data = dict(train=training, validation=development)
    monkeypatch.setattr(module, "load_dataset", lambda path: (new_data, {}, new_manifest) if path == new else (old_data, {}, old_manifest))
    first = np.random.default_rng(seed).permutation(10000)
    prefix = np.concatenate([first, np.arange(10000, 50000)])
    np.save(base / "n10000-float32-state-boxcox/training-indices.npy", first)
    np.save(nested / "training-indices.npy", prefix)
    write(base / "summary.json", dict(dataset_manifest_sha256=sha256(old / "manifest.json"), plan=dict(config=dict(seed=seed))))
    write(audit / "summary.json", dict(status="complete", reference_subset_pass=True, dataset_manifest_sha256=sha256(new / "manifest.json")))
    write(nested / "result.json", dict(status="complete", training_count=50000, config=dict(seed=seed),
          hashes=dict(dataset_manifest="verified-50k-dataset"), artifacts={"training-indices.npy":sha256(nested / "training-indices.npy")}))
    write(nested / "verification.json", dict(status="verified", result_sha256=sha256(nested / "result.json")))
    selected, *_ = module.expanded_inputs(new, audit, old, base, seed, 200000, nested)
    np.testing.assert_array_equal(selected["source_indices"][:50000], prefix)
    assert len(np.unique(selected["source_indices"])) == 200000
    with pytest.raises(ValueError, match="verified nested"):
        module.expanded_inputs(new, audit, old, base, seed, 200000)
    np.save(nested / "training-indices.npy", np.zeros(50000, dtype=int))
    with pytest.raises(ValueError, match="identity"):
        module.expanded_inputs(new, audit, old, base, seed, 200000, nested)
