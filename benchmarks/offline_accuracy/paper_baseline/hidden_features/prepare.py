"""Trusted staging: freeze training features and separate label-free state inputs."""
import json
import shutil
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, digest
from benchmarks.research_contract.replay import population_rows
from ..correction_scaling.model import load_base
from .model import base_and_hidden, make_features, config


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def diversity_indices(count, unseen, requested=50000, prefix=10000):
    unseen = np.asarray(unseen)
    if unseen.ndim != 1 or not np.issubdtype(unseen.dtype, np.integer) or np.any(unseen < prefix) or np.any(unseen >= count):
        raise ValueError("Unseen indices must be valid and outside the original prefix")
    selected = np.setdiff1d(np.arange(count), unseen)[:requested]
    if len(selected) != requested or not np.array_equal(selected[:prefix], np.arange(prefix)):
        raise ValueError("Insufficient disjoint training support")
    return selected


def stage(project, output, bindings, arms=("A", "B", "C")):
    root, transfer = project / bindings["prepared_root"], project / bindings["transfer_root"]
    for seed_text in bindings["seeds"]:
        seed = int(seed_text)
        destination = output / "prepared" / seed_text
        destination.mkdir(parents=True)
        model_dir = destination / "model"
        model_dir.mkdir()
        for name in ("base-weights.pt", "preprocessing.npz"):
            shutil.copyfile(root / seed_text / name, model_dir / name)
        base, prep = load_base(root / seed_text, seed)
        full = arrays(root / seed_text / "training.npz")
        b, h = base_and_hidden(base, prep, full["states"][:50000], seed)
        np.testing.assert_array_equal(b, full["base"][:50000])
        offset, scale = h.astype(np.float64).mean(axis=0), h.astype(np.float64).std(axis=0)
        scale[scale == 0] = 1
        scales = dict(hidden_offset=offset, hidden_scale=scale)
        np.savez(model_dir / "hidden-scales.npz", **scales)
        if "S" in arms and len(arms) != 1:
            raise ValueError("The diversity comparison has one separately registered population")
        indices = (diversity_indices(len(full["states"]), arrays(transfer / "plan/selections.npz")[f"unseen_{seed}"])
                   if "S" in arms else np.arange(10000))
        np.save(destination / "training-indices.npy", indices)
        training = {k:v[indices] for k,v in full.items()}
        for arm in arms:
            path = destination / ("train-" + arm)
            path.mkdir()
            cfg = config(seed, arm)
            x = make_features(training["states"], training["base"], h[:10000] if arm == "B" else None, prep, cfg, scales)
            if arm != "B":
                np.testing.assert_array_equal(x, training["features"])
            np.savez(path / "training.npz", features=x, base=training["base"], delta=training["delta"],
                     target=training["target"], active=prep["active"])
            save(path / "config.json", cfg)
        staged_training = training
        del full, h, b
        for population, rows in population_rows(root, transfer, seed_text):
            if population == "training":
                rows = staged_training
            path = destination / ("input-" + population)
            path.mkdir()
            np.savez(path / "states.npz", states=rows["states"], source_indices=rows["source_indices"])
        save(destination / "manifest.json", dict(seed=seed, calibration_states=50000,
             training_states=len(indices), unseen_overlap=0,
             calibration_labels_used=False, exact_frozen_base_replay=True,
             artifacts={str(p.relative_to(destination)):digest(p) for p in destination.rglob("*") if p.is_file()}))


def verify_stage(path):
    from benchmarks.research_contract.adapter import read_json, checked_file, project_path
    manifest = read_json(path / "manifest.json")
    for name, expected in manifest["artifacts"].items():
        checked_file(project_path(path, name), expected)
    return manifest
