import numpy as np

from benchmarks.flame_conditioning.input_support import profile_population


def test_profile_keeps_raw_ranges_distinct_from_encoded_scales():
    train = np.array([[300., 100., 1e-10], [400., 102., 1e-5]])
    observed = np.array([[350., 90., 0.]])
    offset, scale = np.array([350., 101., .2]), np.array([50., 1., .1])
    rows = profile_population(train, observed, offset, scale, ["T_K", "P_Pa", "X"], "uniform")
    assert rows[0]["outsideTrainingRangeFraction"] == 0
    assert rows[1]["outsideTrainingRangeFraction"] == 1
    assert rows[1]["absoluteStandardizedMax"] == 11
    assert rows[2]["encoding"] == "mass_fraction**0.1"
    assert rows[2]["trainingMin"] == 1e-10
    assert rows[2]["absoluteStandardizedMax"] == 2
    assert rows[2]["observedZeroFraction"] == 1
    assert all(row["population"] == "uniform" and row["samples"] == 1 for row in rows)
