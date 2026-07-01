import time

import numpy as np


def single_step(npstate, chem, time_step=1e-6):
    import cantera as ct

    gas = ct.Solution(chem)
    T_old, P_old, Y_old = npstate[0], npstate[1], npstate[2:]
    gas.TPY = T_old, P_old, Y_old
    res_1st = [T_old, P_old] + list(gas.Y)
    reactor = ct.IdealGasConstPressureReactor(gas, name='R1', clone=False)
    sim = ct.ReactorNet([reactor])

    sim.advance(time_step)
    new_TPY = [gas.T, gas.P] + list(gas.Y)
    res_1st += new_TPY

    return res_1st


def random_perturb(
    array: np.ndarray,
    mech_path: str,
    dataset: int,
    heat_limit: bool,
    element_limit: bool,
    eq_ratio: float = 1,
    frozenTem: float = 310,
    alpha: float = 0.1,
    gamma: float = 0.1,
    cq: float = 10,
    inert_idx: int = -1,
    time_step: float = 1e-6,
    seed: int | None = None,
) -> np.ndarray:
    import cantera as ct

    rng = np.random.default_rng(seed)

    array = array[array[:, 0] > frozenTem]

    gas = ct.Solution(mech_path)
    n_species = gas.n_species
    maxT = np.max(array[:, 0])
    minT = np.min(array[:, 0])
    maxP = np.max(array[:, 1])
    minP = np.min(array[:, 1])
    maxN2 = np.max(array[:, -1])
    minN2 = np.min(array[:, -1])

    H_O_ratio_base = 2 * eq_ratio

    num = 0
    new_array = []
    while num < dataset:
        if heat_limit:
            from dfode_kit.training.formation import formation_calculate

            qdot_ = np.zeros_like(array[:, 0])
            formation = formation_calculate(mech_path)
            label_array = label(array, mech_path)
            for i in range(label_array.shape[0]):
                qdot_[i] = (
                    -(
                        formation
                        * (
                            label_array[i, 4 + n_species : 4 + 2 * n_species]
                            - label_array[i, 2 : 2 + n_species]
                        )
                        / time_step
                    ).sum()
                )

        for j in range(array.shape[0]):
            test_tmp = np.copy(array[j])
            k = 0
            while True:
                k += 1

                test_r = np.copy(array[j])

                test_tmp[0] = test_r[0] + (maxT - minT) * (2 * rng.random() - 1.0) * alpha
                test_tmp[1] = test_r[1] + (maxP - minP) * (2 * rng.random() - 1.0) * alpha * 20
                test_tmp[-1] = test_r[-1] + (maxN2 - minN2) * (2 * rng.random() - 1) * alpha
                for i in range(2, array.shape[1] - 1):
                    test_tmp[i] = np.abs(test_r[i]) ** (1 + (2 * rng.random() - 1) * alpha)
                test_tmp[2:-1] = test_tmp[2:-1] / np.sum(test_tmp[2:-1]) * (1 - test_tmp[-1])

                if heat_limit:
                    label_test_tmp = np.array(single_step(test_tmp, mech_path))
                    qdot_new_ = (
                        -(
                            formation
                            * (
                                label_test_tmp[4 + n_species : 4 + 2 * n_species]
                                - label_test_tmp[2 : 2 + n_species]
                            )
                            / time_step
                        ).sum()
                    )

                if element_limit:
                    gas.TPY = test_tmp[0], test_tmp[1], test_tmp[2:]
                    H_mole_fraction = gas.elemental_mole_fraction("H")
                    O_mole_fraction = gas.elemental_mole_fraction("O")
                    H_O_ratio = H_mole_fraction / O_mole_fraction

                if heat_limit and element_limit:
                    condition = (
                        (minT * (1 - gamma)) <= test_tmp[0] <= (maxT * (1 + gamma))
                        and (H_O_ratio_base * (1 - gamma)) <= H_O_ratio <= (H_O_ratio_base * (1 + gamma))
                        and (qdot_new_ > 1 / cq * qdot_[j] and qdot_new_ < cq * qdot_[j])
                    )
                elif heat_limit and not element_limit:
                    condition = (
                        (minT * (1 - gamma)) <= test_tmp[0] <= (maxT * (1 + gamma))
                        and (qdot_new_ > 1 / cq * qdot_[j] and qdot_new_ < cq * qdot_[j])
                    )
                elif not heat_limit and element_limit:
                    condition = (
                        (minT * (1 - gamma)) <= test_tmp[0] <= (maxT * (1 + gamma))
                        and (H_O_ratio_base * (1 - gamma)) <= H_O_ratio <= (H_O_ratio_base * (1 + gamma))
                    )
                else:
                    condition = (minT * (1 - gamma)) <= test_tmp[0] <= (maxT * (1 + gamma))

                if condition or k > 20:
                    break

            if k <= 20:
                new_array.append(test_tmp)

        num = len(new_array)
        print(num)

    new_array = np.array(new_array)
    new_array = new_array[rng.choice(new_array.shape[0], size=dataset)]
    unique_array = np.unique(new_array, axis=0)
    print(unique_array.shape)
    return unique_array


def label(
    array: np.ndarray,
    mech_path: str,
    time_step: float = 1e-06,
) -> np.ndarray:
    import cantera as ct

    from dfode_kit.data.integration import advance_reactor

    gas = ct.Solution(mech_path)
    n_species = gas.n_species

    labeled_data = np.empty((array.shape[0], 2 * n_species + 4))

    reactor = ct.Reactor(gas, name='Reactor1', energy='off', clone=False)
    reactor_net = ct.ReactorNet([reactor])
    reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10

    start_time = time.time()

    for i, state in enumerate(array):
        gas = advance_reactor(gas, state, reactor, reactor_net, time_step)
        labeled_data[i, : 2 + n_species] = state[: 2 + n_species]
        labeled_data[i, 2 + n_species :] = np.array([gas.T, gas.P] + list(gas.Y))

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time used: {total_time:.2f} seconds")

    return labeled_data
