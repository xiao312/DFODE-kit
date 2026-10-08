import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.heldout import normalize_rows, sample_cells


def test_sample_preserves_uniform_population_and_distinct_cell_ids():
    temperature = np.linspace(300, 2200, 5000)
    ids, uniform, balanced, populations = sample_cells(temperature)
    assert len(ids) == len(set(ids))
    assert uniform.sum() == 1024
    assert balanced.sum() == sum(min(32, item["cells"]) for item in populations)
    assert sum(item["cells"] for item in populations) == 5000
    np.testing.assert_array_equal(ids, sample_cells(temperature)[0])


def test_normalization_records_closure_without_negative_clipping():
    raw = np.array([[300, 101325, .6, .400001]])
    states, mass_sum = normalize_rows(raw, 2)
    assert mass_sum[0] == pytest.approx(1.000001)
    assert states[0, 2:].sum() == pytest.approx(1)
    assert raw[0, 3] == .400001
    with pytest.raises(ValueError):
        normalize_rows(np.array([[300, 101325, 1.000001, -.000001]]), 2)


def test_hybrid_uses_fixed_thresholds_and_cold_zero():
    pytest.importorskip("torch")
    from benchmarks.flame_conditioning.evaluate_heldout import hybrid_prediction
    states = np.column_stack([np.array([300, 305, 999, 1000]), np.ones(4), np.ones(4)])
    def boxcox(rows):
        return np.full((len(rows), 1), 2.), np.ones((len(rows), 1), dtype=bool)
    def power(rows):
        return np.ones((len(rows), 1)), np.zeros((len(rows), 1), dtype=bool)
    predicted, corrected = hybrid_prediction(states, boxcox, power)
    np.testing.assert_array_equal(predicted[:, 0], [0, 1, 1, 2])
    np.testing.assert_array_equal(corrected[:, 0], [False, False, False, True])
