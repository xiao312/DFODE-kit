"""State-only kinetic features, fitting-only scaling and shared correction decode."""
import numpy as np
from benchmarks.flame_conditioning.chemistry import set_state
from benchmarks.offline_accuracy.improve.physics_prior import frozen_step
from ..adaptive.model import features, base_preprocessing
from ..correction_scaling.model import base_prediction, cached_prediction


def rate_features(gas, states, active):
    """One initial kinetic evaluation per state; no time integration or labels."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != gas.n_species + 2:
        raise ValueError("Expected full ordered T/P/Y states")
    gas.basis = "mass"
    values = []
    for row in states:
        if not np.isfinite(row).all():
            raise ValueError("Nonfinite physical state")
        set_state(gas, dict(T=row[0], P=row[1], Y=row[2:]))
        conversion = gas.molecular_weights / gas.density_mass
        net = gas.net_production_rates * conversion
        destruction = gas.destruction_rates * conversion
        euler = 1e-6 * net
        frozen = frozen_step(row[2:], net, destruction, 1e-6, "frozen-exponential")
        values.append(np.r_[euler[active], frozen[active]])
    raw = np.asarray(values)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite initial-rate features; no clipping fallback")
    return np.arcsinh(raw / 1e-14), raw


def fit_scaler(values):
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid fitting features")
    return dict(mean=values.mean(0), scale=np.maximum(values.std(0), 1e-6))


def augment(ordinary, rates, scaler):
    if len(ordinary) != len(rates) or rates.shape[1:] != scaler["mean"].shape:
        raise ValueError("Feature axes differ")
    if np.any(scaler["scale"] <= 0):
        raise ValueError("Invalid feature scale")
    result = np.concatenate([ordinary, (rates-scaler["mean"])/scaler["scale"]], axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prepared features")
    return result


def predict(base, head, prep, states, config, gas=None, scaler=None):
    b, _ = base_prediction(base, base_preprocessing(prep), states, config["seed"])
    x = features(states, b, prep, config).astype(np.float32)
    if config["arm"] == "kinetic":
        if gas is None or scaler is None:
            raise ValueError("Kinetic model requires mechanism and frozen scaler")
        x = augment(x, rate_features(gas, states, prep["active"])[0], scaler)
    return cached_prediction(head, dict(base=b, features=x), prep["active"], config)
