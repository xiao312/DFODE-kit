from dfode_kit.data.sequence_suites import split_conditions, suite_cases
from dfode_kit.data.sequences import build_reactor_conditions


def test_suite_cases_include_h2_and_ch4_pilot_cases():
    cases = suite_cases("pilot")
    ids = {case.case_id for case in cases}

    assert "burke_h2_logtime" in ids
    assert "h2o2_h2_logtime" in ids
    assert "gri30_ch4_logtime" in ids


def test_reactive_pilot_uses_longer_horizons_for_reactivity():
    cases = {case.case_id: case for case in suite_cases("reactive-pilot")}

    assert cases["burke_h2_reactive_logtime"].t_end == 1e-4
    assert cases["h2o2_h2_reactive_logtime"].t_end == 1e-4
    assert cases["gri30_ch4_reactive_logtime"].t_end == 1e-2


def test_split_conditions_holds_out_whole_conditions():
    conditions = build_reactor_conditions(
        temperatures=[900, 1000, 1100, 1200],
        pressures=[101325],
        phis=[0.6, 1.0],
        fuel="H2:1.0",
        oxidizer="O2:1.0,N2:3.76",
    )

    train, val = split_conditions(conditions, val_fraction=0.25, seed=7)

    assert len(train) == 6
    assert len(val) == 2
    assert set(train).isdisjoint(set(val))
