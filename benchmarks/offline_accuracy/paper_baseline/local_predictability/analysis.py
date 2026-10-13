"""Trusted diagnostic calculations on preselected graphs; never candidate inference."""
import numpy as np
from benchmarks.research_contract.reference import evaluate_predictions, _quantiles
from ..decoder_loss.coordinates import decode

SCALE = 1e-14


def stratified_rows(rows, per_snapshot):
    selected = []
    rng = np.random.default_rng(20261013)
    for source in np.unique(rows["snapshot"]):
        ix = np.flatnonzero(rows["snapshot"] == source)
        ix = ix[np.argsort(rows["source_indices"][ix], kind="stable")]
        if len(ix) < per_snapshot:
            raise ValueError("Insufficient snapshot support")
        selected.extend(np.sort(rng.choice(ix,per_snapshot,replace=False)))
    return np.asarray(selected, dtype=int)


def transfer(library, query_base, indices, active, mode):
    z = np.arcsinh(library["delta"][:, active] / SCALE)
    if mode == "correction":
        values = z-np.arcsinh(library["base"][:, active]/SCALE)
        coordinate = values[indices].mean(axis=1)
        pred = decode(query_base[:, active], coordinate, dict(decoder="asinh", scale_floor=SCALE))
    elif mode == "target":
        coordinate = z[indices].mean(axis=1)
        pred = SCALE*np.sinh(coordinate)
    else:
        raise ValueError("Unknown diagnostic mode")
    output = np.zeros_like(query_base)
    output[:, active] = pred
    return output, coordinate


def summarize(prediction, coordinate, query, active, species, mode):
    report = evaluate_predictions(prediction, query["delta"], query["base"], species,
                                  initial=query["states"][:,2:])
    # Independent count uses no scorer grouping code.
    error = abs(prediction[:,active]-query["delta"][:,active])/(1e-15+.1*abs(query["delta"][:,active]))
    if int((error<=1).sum()) != report["groups"]["all"]["pass_count"]:
        raise ValueError("Physical pass recount differs")
    wanted = np.arcsinh(query["delta"][:,active]/SCALE)
    if mode == "correction":
        wanted -= np.arcsinh(query["base"][:,active]/SCALE)
    difference = abs(coordinate-wanted)
    report["coordinate_error"] = dict(quantiles=_quantiles(difference),
        mean=float(difference.mean()), rmse=float(np.sqrt(np.mean(difference**2))),
        within_sufficient_band=float((difference<=np.log1p(.1)).mean()))
    report["by_snapshot"] = {}
    for snapshot in np.unique(query["snapshot"]):
        mask = query["snapshot"] == snapshot
        report["by_snapshot"][str(snapshot)] = dict(states=int(mask.sum()), acceptance=float((error[mask]<=1).mean()))
    return report
