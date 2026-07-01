import h5py
import numpy as np

from dfode_kit.data.contracts import MECHANISM_ATTR, require_h5_attr, read_scalar_field_datasets
from dfode_kit.data.io_hdf5 import get_TPY_from_h5, touch_h5
from dfode_kit.utils import BCT, inverse_BCT


def advance_reactor(gas, state, reactor, reactor_net, time_step):
    """Advance the reactor simulation for a given state."""
    state = state.flatten()

    expected_shape = (2 + gas.n_species,)
    if state.shape != expected_shape:
        raise ValueError(
            f"Expected state shape {expected_shape}, got {state.shape}"
        )

    gas.TPY = state[0], state[1], state[2:]

    reactor.syncState()
    reactor_net.reinitialize()
    reactor_net.advance(time_step)
    reactor_net.set_initial_time(0.0)

    return gas


def load_model(model_path, device, model_class, model_layers):
    import torch

    state_dict = torch.load(model_path, map_location='cpu')

    model = model_class(model_layers)
    model.load_state_dict(state_dict['net'])

    model.eval()
    model.to(device=device)

    return model


def predict_Y(model, model_path, d_arr, mech, device):
    import torch
    import cantera as ct

    gas = ct.Solution(mech)
    n_species = gas.n_species
    expected_dims = 2 + n_species
    if d_arr.shape[1] != expected_dims:
        raise ValueError(
            f"Expected input with {expected_dims} columns, got {d_arr.shape[1]}"
        )

    state_dict = torch.load(model_path, map_location='cpu')

    Xmu0 = state_dict['data_in_mean']
    Xstd0 = state_dict['data_in_std']
    Ymu0 = state_dict['data_target_mean']
    Ystd0 = state_dict['data_target_std']

    d_arr = np.clip(d_arr, 0, None)
    d_arr[:, 1] *= 0
    d_arr[:, 1] += 101325

    orig_Y = d_arr[:, 2:].copy()
    in_bct = d_arr.copy()
    in_bct[:, 2:] = BCT(in_bct[:, 2:])
    in_bct_norm = (in_bct - Xmu0) / Xstd0

    input = torch.from_numpy(in_bct_norm).float().to(device=device)

    output = model(input)

    out_bct = output.cpu().numpy() * Ystd0 + Ymu0 + in_bct[:, 2:-1]
    next_Y = orig_Y.copy()
    next_Y[:, :-1] = inverse_BCT(out_bct)
    next_Y[:, :-1] = next_Y[:, :-1] / np.sum(next_Y[:, :-1], axis=1, keepdims=True) * (1 - next_Y[:, -1:])

    return next_Y


def nn_integrate(orig_arr, model_path, device, model_class, model_layers, time_step, mech, frozen_temperature=510):
    import cantera as ct

    model = load_model(model_path, device, model_class, model_layers)

    mask = orig_arr[:, 0] > frozen_temperature
    infer_arr = orig_arr[mask, :]

    next_Y = predict_Y(model, model_path, infer_arr, mech, device)

    new_states = np.hstack((np.zeros((orig_arr.shape[0], 1)), orig_arr))
    new_states[:, 0] += time_step
    new_states[:, 2] = orig_arr[:, 1]
    new_states[mask, 3:] = next_Y

    setter_gas = ct.Solution(mech)
    getter_gas = ct.Solution(mech)
    new_T = np.zeros_like(next_Y[:, 0])

    for idx, (state, next_y) in enumerate(zip(infer_arr, next_Y)):
        try:
            setter_gas.TPY = state[0], state[1], state[2:]
            h = setter_gas.enthalpy_mass

            getter_gas.Y = next_y
            getter_gas.HP = h, state[1]

            new_T[idx] = getter_gas.T

        except ct.CanteraError:
            continue
    new_states[mask, 1] = new_T

    return new_states


def integrate_h5(
    file_path,
    save_path1,
    save_path2,
    time_step,
    cvode_integration=True,
    nn_integration=False,
    model_settings=None,
):
    import cantera as ct

    """Process scalar-field datasets and save CVODE / NN integration outputs."""
    with h5py.File(file_path, 'r') as f:
        mech = require_h5_attr(f, MECHANISM_ATTR)

    data_dict = read_scalar_field_datasets(file_path)

    if cvode_integration:
        gas = ct.Solution(mech)
        reactor = ct.Reactor(gas, name='Reactor1', energy='off', clone=False)
        reactor_net = ct.ReactorNet([reactor])
        reactor_net.rtol, reactor_net.atol = 1e-6, 1e-10

        processed_data_dict = {}

        for name, data in data_dict.items():
            processed_data = np.empty((data.shape[0], data.shape[1] + 1))
            for i, state in enumerate(data):
                gas = advance_reactor(gas, state, reactor, reactor_net, time_step)

                new_state = np.array([time_step, gas.T, gas.P] + list(gas.Y))

                processed_data[i, :] = new_state

            processed_data_dict[name] = processed_data

        with h5py.File(save_path1, 'a') as f:
            cvode_group = f.create_group('cvode_integration')

            for dataset_name, processed_data in processed_data_dict.items():
                cvode_group.create_dataset(dataset_name, data=processed_data)
                print(f'Saved processed dataset: {dataset_name} in cvode_integration group')

    if nn_integration:
        processed_data_dict = {}
        if model_settings is None:
            raise ValueError("model_settings must be provided for neural network integration.")

        for name, data in data_dict.items():
            try:
                processed_data = nn_integrate(data, **model_settings)
                processed_data_dict[name] = processed_data
            except Exception as e:
                print(f"Error processing dataset '{name}': {e}")

        with h5py.File(save_path2, 'a') as f:
            if 'nn_integration' in f:
                del f['nn_integration']
            nn_group = f.create_group('nn_integration')

            for dataset_name, processed_data in processed_data_dict.items():
                nn_group.create_dataset(dataset_name, data=processed_data)
                print(f'Saved processed dataset: {dataset_name} in nn_integration group')


def calculate_error(
    mech_path,
    save_path1,
    save_path2,
    error='RMSE'
):
    import cantera as ct

    gas = ct.Solution(mech_path)

    with h5py.File(save_path1, 'r') as f1, h5py.File(save_path2, 'r') as f2:
        cvode_group = f1['cvode_integration']
        nn_group = f2['nn_integration']

        common_datasets = set(cvode_group.keys()) & set(nn_group.keys())

        sorted_datasets = sorted(common_datasets, key=lambda x: float(x))
        results = {}

        for ds_name in sorted_datasets:
            cvode_data = cvode_group[ds_name][:, 3:]
            nn_data = nn_group[ds_name][:, 3:]

            if error == "RMSE":
                rmse_per_dim = np.sqrt(np.mean((cvode_data - nn_data) ** 2, axis=0))
                results[ds_name] = rmse_per_dim

                print(f"RMSE of ataset: {ds_name}")
                for dim_idx, rmse_val in enumerate(rmse_per_dim, start=1):
                    id = gas.species_names[dim_idx - 3]
                    print(f"  Species {id}: {rmse_val:.6e}")
                print()

    return results
