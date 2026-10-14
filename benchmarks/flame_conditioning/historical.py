"""Historical-weight controls from checked numerical arrays, never old pickle."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.flame_conditioning.coordinates import inverse_state_change
from benchmarks.flame_conditioning.extract import sha256

MODES = ("source-formula", "stable-adapter")
SOURCE_HASHES = {
    "state-boxcox": "7938fc47584da9cad2b811e9607ed6413bf210d3ae4c20f3dca64807ae56abee",
    "signed-power": "8ed4105be6eab440b57141072dc384a4df2454d7570df5a7647a10599a471089",
}


def transformed_inputs(states, kind):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 61 or not np.isfinite(states).all():
        raise ValueError("Expected finite T, p, and 59 mechanism-ordered fractions")
    if np.any(states[:, :2] <= 0) or np.any(states[:, 2:] < 0):
        raise ValueError("Invalid physical inputs; do not silently repair them")
    values = np.abs(states) if kind == "state-boxcox" else states.copy()
    if kind == "signed-power":
        values[:, 2:] = np.clip(values[:, 2:], 0, 1)
    values[:, 2:] = (values[:, 2:] ** .1 - 1) / .1
    return values


def reconstruct(states, standardized_inputs, network_output, arrays, kind, mode):
    """Return delta, explicit repairs, and raw inverse-domain violations."""
    if mode not in MODES or kind not in SOURCE_HASHES:
        raise ValueError("Unknown historical reconstruction contract")
    precision = np.float32 if mode == "source-formula" else np.float64
    encoded = (network_output.astype(precision) * arrays["labels_std"].astype(precision)
               + arrays["labels_mean"].astype(precision))
    correction = np.zeros((len(states), 59), dtype=bool)
    violations = np.zeros((len(states), 58), dtype=bool)
    if kind == "signed-power":
        # The original direct-power target has NO division by lambda.
        active_delta = np.sign(encoded) * np.abs(encoded) ** 10
    elif mode == "stable-adapter":
        active_delta, correction[:, :58] = inverse_state_change(states[:, 2:60], encoded)
        violations = correction[:, :58].copy()
    else:
        initial_bc = (standardized_inputs.astype(np.float32) * arrays["features_std"].astype(np.float32)
                      + arrays["features_mean"].astype(np.float32))[:, 2:60]
        initial_base = .1 * initial_bc + 1
        final_base = .1 * (initial_bc + encoded) + 1
        violations = final_base < 0
        # Reproduce the inspected source formula, including its unguarded base.
        # This is a control, not an endorsed physical repair or deployment path.
        active_delta = final_base ** 10 - initial_base ** 10
    delta = np.zeros((len(states), 59), dtype=np.float64)
    delta[:, :58] = active_delta
    if not np.isfinite(delta).all():
        raise ValueError("Historical reconstruction produced nonfinite increments")
    return delta, correction, violations


def load_historical(directory, species_names, mode):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    kind = manifest["kind"]
    if (manifest["status"] != "converted" or kind not in SOURCE_HASHES
            or manifest["checkpoint_sha256"] != SOURCE_HASHES[kind]
            or manifest["species_names"] != species_names or species_names[-1] != "AR"
            or sha256(directory / "model.npz") != manifest["arrays_sha256"]):
        raise ValueError("Historical numerical artifact contract mismatch")
    with np.load(directory / "model.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    sizes = [61, 800, 800, 800, 800, 58]
    layers = []
    for index in range(5):
        layer = torch.nn.Linear(sizes[index], sizes[index + 1], dtype=torch.float32)
        layer.load_state_dict({"weight": torch.from_numpy(arrays[f"layer{index}_weight"]),
                               "bias": torch.from_numpy(arrays[f"layer{index}_bias"])})
        layers.append(layer)
        if index < 4:
            layers.append(torch.nn.GELU())
    model = torch.nn.Sequential(*layers).eval()
    diagnostics = {}

    def predict(states):
        values = transformed_inputs(states, kind)
        # Preserve saved precision, but do the NumPy input calculation in FP64.
        standardized = (values - arrays["features_mean"]) / arrays["features_std"]
        standardized = standardized.astype(np.float32)
        with torch.no_grad():
            output = model(torch.from_numpy(standardized)).numpy()
        delta, correction, violations = reconstruct(states, standardized, output, arrays, kind, mode)
        diagnostics.update(inverse_domain_violation_fraction=float(violations.mean()),
                           reconstruction=mode, historical_training_overlap_not_excluded=True)
        return delta, correction

    predict.diagnostics = diagnostics
    return predict, manifest
