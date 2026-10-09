"""Training-only local RBF comparator, not an error-certified ISAT table."""
import time

import numpy as np
from scipy.interpolate import RBFInterpolator

from benchmarks.flame_conditioning.coordinates import input_features, state_change, inverse_state_change


class Predictor:
    def __init__(self, arrays, config):
        self.arrays, self.config = arrays, config
        self.model = RBFInterpolator(arrays["points"], arrays["targets"],
            neighbors=min(config["neighbors"], len(arrays["points"])), smoothing=config["smoothing"],
            kernel="cubic", degree=1)

    def __call__(self, states):
        a = self.arrays
        x = (input_features(states)-a["x_offset"])/a["x_scale"]
        coordinate = self.model(x @ a["basis"]) * a["y_scale"]
        if self.config["name"] == "local-state":
            value, corrected = inverse_state_change(states[:, 2:], coordinate)
        else:
            value = 1e-14 * np.sinh(coordinate)
            corrected = np.zeros_like(value, dtype=bool)
        value[:, ~a["active"]], corrected[:, ~a["active"]] = 0., False
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite local prediction")
        return value, corrected


def fit(training, config, preprocessing, destination):
    wall, cpu = time.monotonic(), time.process_time()
    p = preprocessing
    x = (input_features(training["states"])-p["x_offset"])/p["x_scale"]
    _, singular, vh = np.linalg.svd(x, full_matrices=False)
    basis = vh[singular > singular[0]*1e-10].T
    if basis.shape[1]+1 >= min(config["neighbors"], len(x)):
        raise ValueError("Too few neighbors for local linear polynomial")
    truth = training["delta"]
    target = (state_change(training["states"][:, 2:], truth) if config["name"] == "local-state"
              else np.arcsinh(truth/1e-14))
    scale = np.maximum(np.sqrt(np.mean(target**2, axis=0)), 1e-30)
    arrays = dict(points=x @ basis, targets=target/scale, basis=basis,
                  y_scale=scale, x_offset=p["x_offset"], x_scale=p["x_scale"], active=p["active"])
    predictor = Predictor(arrays, config)
    np.savez_compressed(destination / "local-table.npz", **arrays)
    return predictor, dict(history=[], updates_completed=0, numerical_input_rank=basis.shape[1],
                           parameter_count=None, table_bytes=sum(a.nbytes for a in arrays.values()),
                           fit_process_seconds=time.process_time()-cpu, fit_wall_seconds=time.monotonic()-wall)


def reload(directory, config):
    with np.load(directory / "local-table.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    return Predictor(arrays, config)
