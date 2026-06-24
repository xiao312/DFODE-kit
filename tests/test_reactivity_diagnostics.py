import numpy as np

from dfode_kit.data.reactivity import reactive_step_mask, summarize_trajectory_reactivity


def test_reactive_step_mask_detects_temperature_and_species_changes():
    states = np.array(
        [
            [1000.0, 101325.0, 1.0, 0.0],
            [1000.2, 101325.0, 1.0, 0.0],
            [1002.0, 101325.0, 0.999, 0.001],
        ]
    )

    mask = reactive_step_mask(states, step_temperature_delta=1.0, step_species_delta=1e-4)

    assert mask.tolist() == [False, True]


def test_reactivity_summary_reports_reactive_fraction():
    states = np.array(
        [
            [1000.0, 101325.0, 1.0, 0.0],
            [1001.5, 101325.0, 0.999, 0.001],
            [1003.0, 101325.0, 0.998, 0.002],
        ]
    )
    times = np.array([0.0, 1e-6, 2e-6])

    summary = summarize_trajectory_reactivity(states, times, step_temperature_delta=1.0)

    assert summary["reactive_step_fraction"] == 1.0
    assert summary["is_reactive"] is True
