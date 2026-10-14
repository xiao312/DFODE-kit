import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.filter_audit import curation_mask, population_record


def test_rule_keeps_mild_endothermic_change_and_exact_boundary():
    delta = np.array([[-.1], [.0001], [.002], [.00201]])
    kept, energy = curation_mask(delta, np.array([100000.]))
    np.testing.assert_array_equal(kept, [True, True, True, False])
    np.testing.assert_allclose(energy, [-10000, 10, 200, 201])
    record = population_record(np.zeros((4, 3)), kept, energy)
    assert record["rejected_fraction"] == .25
    assert record["endothermic_fraction"] == .75
    assert population_record(np.zeros((0, 3)), kept[:0], energy[:0])["rejected_fraction"] is None
