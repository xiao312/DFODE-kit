"""Train-only residual scale calibration; inference receives only base predictions."""
import numpy as np
from ..coordinates import standardize


def magnitude_coordinate(base, config):
    return np.log1p(np.abs(base)/config["scale_floor"])/np.log(10.)


def calibrate(base, truth, config):
    if base.shape != truth.shape or base.ndim != 2 or not len(base) or not np.isfinite([base, truth]).all():
        raise ValueError("Require finite aligned calibration arrays")
    edges = np.asarray(config["bin_edges"], dtype=float)
    position = magnitude_coordinate(base, config)
    ratio = np.abs(truth-base)/(config["scale_floor"]+np.abs(base))
    global_scale = np.quantile(ratio, config["calibration_quantile"], axis=0)
    raw = np.zeros((base.shape[1],len(edges)-1))
    counts = np.zeros_like(raw, dtype=np.int64)
    # Include out-of-range values in edge bins; interpolation has constant tails.
    bins = np.clip(np.searchsorted(edges, position, side="right")-1, 0, len(edges)-2)
    for species in range(base.shape[1]):
        for index in range(len(edges)-1):
            selected = bins[:,species] == index
            counts[species,index] = selected.sum()
            raw[species,index] = (np.quantile(ratio[selected,species],config["calibration_quantile"])
                if selected.sum() >= config["minimum_bin_count"] else global_scale[species])
    return dict(alpha_raw=raw, alpha=np.clip(raw,config["alpha_min"],config["alpha_max"]),
                calibration_counts=counts, alpha_global=global_scale, bin_centers=(edges[:-1]+edges[1:])/2)


def scales(base, prep, config):
    scale = config["scale_floor"]+np.abs(base)
    if config["arm"] == "calibrated":
        position = magnitude_coordinate(base, config)
        factors = np.column_stack([np.exp(np.interp(position[:,j],prep["bin_centers"],np.log(prep["alpha"][j])))
                                   for j in range(base.shape[1])])
        scale = scale*factors
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid inference scale")
    return scale


def preprocessing(base_prep, base, truth, config):
    active = base_prep["active"]
    extra = np.arcsinh(base[:,active]/config["scale_floor"])
    offset, scale, constant = standardize(extra,1)
    return dict({"base_"+key:value for key,value in base_prep.items()}, active=active,
        extra_offset=offset, extra_scale=scale, extra_constant=constant,
        **calibrate(base[:,active],truth[:,active],config))


def residual_roundtrip(base, truth):
    reconstructed = base+(truth-base)
    ratio = abs(reconstructed-truth)/(1e-15+.1*abs(truth))
    return dict(max_error_over_allowance=float(ratio.max()), exact_fraction=float((reconstructed==truth).mean()))
