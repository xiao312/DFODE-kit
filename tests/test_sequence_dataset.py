import numpy as np

from dfode_kit.data.sequences import ReactorCondition, build_reactor_conditions, fixed_time_grid, log_time_grid


def test_build_reactor_conditions_cross_product():
    conditions = build_reactor_conditions(
        temperatures=[900.0, 1000.0],
        pressures=[101325.0],
        phis=[0.6, 1.0],
        fuel="H2:1.0",
        oxidizer="O2:1.0,N2:3.76",
    )

    assert len(conditions) == 4
    assert conditions[0] == ReactorCondition(
        temperature=900.0,
        pressure=101325.0,
        phi=0.6,
        fuel="H2:1.0",
        oxidizer="O2:1.0,N2:3.76",
    )


def test_time_grids_include_zero_and_expected_length():
    fixed = fixed_time_grid(dt=1e-6, steps=3)
    logged = log_time_grid(t_end=1e-3, steps=3, t_start=1e-7)

    assert np.allclose(fixed, [0.0, 1e-6, 2e-6, 3e-6])
    assert logged.shape == (4,)
    assert logged[0] == 0.0
    assert np.all(np.diff(logged) > 0)
