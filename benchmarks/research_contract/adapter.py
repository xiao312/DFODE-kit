"""Identity checks and fixed-contract scoring; never infer alignment from shape."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .reference import evaluate_predictions, _group


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_file(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected:
        raise ValueError(f"Pinned artifact changed: {path}")
    return Path(path)


def project_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Artifact must stay inside the declared project")
    return path


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def checked_artifacts(directory, manifest):
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name:
            raise ValueError("Manifest artifact names must be plain filenames")
        checked_file(directory / name, expected)


def aligned_score(predictions, rows, species, *, prediction_species, population_id,
                  prediction_population_id, baseline=None):
    if population_id != prediction_population_id or list(species) != list(prediction_species):
        raise ValueError("Population or ordered species identity differs")
    expected, actual = rows["source_indices"], predictions["source_indices"]
    if (expected.ndim != 1 or len(np.unique(expected)) != len(expected)
            or not np.array_equal(actual, expected)):
        raise ValueError("Ordered row identities differ or contain duplicates")
    base = rows["base"] if baseline is None else baseline
    scored = evaluate_predictions(predictions["prediction"], rows["delta"], base, species,
                                  initial=rows["states"][:, 2:], atol=1e-15, rtol=.1)
    scored["population_id"] = population_id
    scored["reference_quality"] = "Nominal; only separately audited components have empirical qualification"
    scored["row_identity_sha256"] = hashlib.sha256(expected.astype("<i8").tobytes()).hexdigest()
    active = np.array([name != "AR" for name in species])
    truth = rows["delta"][:, active]
    with np.errstate(over="ignore", invalid="ignore"):
        error = abs(predictions["prediction"][:, active] - truth) / (1e-15 + .1 * abs(truth))
        old = abs(base[:, active] - truth) / (1e-15 + .1 * abs(truth))
    error = np.where(np.isfinite(error), error, np.inf)
    slices = {}
    for snapshot in np.unique(rows["snapshot"]):
        slices["snapshot:" + str(snapshot)] = np.broadcast_to(
            (rows["snapshot"] == snapshot)[:, None], truth.shape)
    for label, mask in (("negative", truth < 0), ("zero", truth == 0), ("positive", truth > 0)):
        slices["sign:" + label] = mask
    bounds = [0, 305, 500, 1000, 1500, 2000, np.inf]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        slices[f"temperature:[{lo},{hi})"] = np.broadcast_to(
            ((rows["states"][:, 0] >= lo) & (rows["states"][:, 0] < hi))[:, None], truth.shape)
    edges = np.array([1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, np.inf])
    bins = np.where(truth == 0, 0, 1 + np.searchsorted(edges, abs(truth), side="left"))
    scored["historical_magnitude_bins"] = {
        str(i): _group(error, old, bins == i) for i in range(9)}
    scored["species_magnitude"] = {}
    for j, name in enumerate(scored["scored_species"]):
        for i in range(9):
            mask = bins[:, j] == i
            scored["species_magnitude"][f"{name}:{i}"] = _group(error[:, j], old[:, j], mask)
    scored["slices"] = {name: _group(error, old, mask) for name, mask in slices.items()}
    return scored
