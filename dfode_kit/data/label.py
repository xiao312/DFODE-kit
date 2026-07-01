import time

import numpy as np


def label_npy(
    mech_path,
    time_step,
    source_path,
):
    import cantera as ct

    from dfode_kit.data.integration import advance_reactor

    gas = ct.Solution(mech_path)
    n_species = gas.n_species

    test_data = np.load(source_path)
    print(f"Loaded dataset from: {source_path}")
    print(f"{test_data.shape=}")

    labeled_data = np.empty((test_data.shape[0], 2 * n_species + 4))

    reactor = ct.Reactor(gas, name='Reactor1', energy='off', clone=False)
    reactor_net = ct.ReactorNet([reactor])
    reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10

    start_time = time.time()

    for i, state in enumerate(test_data):
        gas = advance_reactor(gas, state, reactor, reactor_net, time_step)
        labeled_data[i, : 2 + n_species] = state[: 2 + n_species]
        labeled_data[i, 2 + n_species :] = np.array([gas.T, gas.P] + list(gas.Y))

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time used: {total_time:.2f} seconds")

    return labeled_data
