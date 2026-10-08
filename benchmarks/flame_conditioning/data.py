"""Read a completed dataset, verify hashes, and expose no 2D test interface."""
from __future__ import annotations

import json
from pathlib import Path

import cantera as ct
import numpy as np

from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.extract import sha256


def load_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_exclusions"):
        raise ValueError("Dataset is not complete; do not train on an unfinished run")
    mechanism = root / "mechanism.yaml"
    if sha256(mechanism) != manifest["source_manifest"]["mechanism_sha256"]:
        raise ValueError("Mechanism hash mismatch")
    gas = ct.Solution(str(mechanism))
    species = manifest["source_manifest"]["species_names"]
    if gas.species_names != species:
        raise ValueError("Species order mismatch")
    loaded = {}
    for split in ("train", "validation"):
        directory = root / split
        for filename in ("inputs", "labels"):
            if sha256(directory / f"{filename}.npz") != manifest["splits"][split][f"{filename}_sha256"]:
                raise ValueError(f"{split} {filename} hash mismatch")
        inputs = np.load(directory / "inputs.npz", allow_pickle=False)
        labels = np.load(directory / "labels.npz", allow_pickle=False)
        mask = labels["accepted"]
        if mask.dtype != bool or mask.shape != (len(inputs["states"]),) or mask.mean() < .9:
            raise ValueError("Require at least 90% valid labels in each complete split")
        states, delta = inputs["states"][mask], labels["delta"][mask]
        if states.shape != (len(delta), len(species) + 2) or delta.shape[1] != len(species):
            raise ValueError("Invalid data shape")
        if not np.isfinite(states).all() or not np.isfinite(delta).all() or np.any(states[:, 2:] + delta < 0):
            raise ValueError("Accepted data contain invalid states or labels")
        loaded[split] = {"states": states, "delta": delta, "source_indices": np.flatnonzero(mask),
                         "snapshot": inputs["snapshot"][mask]}
    if set(loaded["train"]["snapshot"]) & set(loaded["validation"]["snapshot"]):
        raise ValueError("Snapshot leakage between training and validation")
    gas.TP = 298.15, ct.one_atm
    physics = {"species_names": species, "element_matrix": element_matrix(gas),
               "formation_enthalpies": gas.standard_enthalpies_RT * ct.gas_constant * gas.T / gas.molecular_weights,
               "molecular_weights": gas.molecular_weights, "interval": manifest["config"]["interval_s"]}
    return loaded, physics, manifest
