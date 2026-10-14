"""Reference-only offline scorer for DFODE increment research.

No files, jobs, network, model imports or promotion writes are performed.
The repository adapter must establish row/species alignment and protected provenance.
A failed/nonfinite candidate is counted, never silently removed from denominators.
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np


def _array(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} shape {out.shape} differs from {shape}")
    return out


def _fraction(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    """Linear empirical quantiles; null means empty or a nonfinite quantile.

    Nonfinite errors remain in the sorted population as +inf. They are not
    discarded before calculating percentiles. The report explains null values.
    """
    flat = np.sort(np.asarray(values, dtype=float).ravel())
    result: dict[str, float | None] = {}
    for name, q in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.)):
        if not len(flat):
            result[name] = None
            continue
        pos = (len(flat) - 1) * q
        lower, upper = int(np.floor(pos)), int(np.ceil(pos))
        if not np.isfinite(flat[lower]) or not np.isfinite(flat[upper]):
            result[name] = None
        else:
            result[name] = float(flat[lower] + (pos-lower) * (flat[upper]-flat[lower]))
    return result


def _group(error: np.ndarray, baseline_error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    e, b = error[mask], baseline_error[mask]
    passed, old = e <= 1., b <= 1.
    n = int(mask.sum())
    p, p0 = int(passed.sum()), int(old.sum())
    repair = int((~old & passed).sum())
    damage = int((old & ~passed).sum())
    return {
        "components": n,
        "pass_count": p,
        "acceptance": _fraction(p, n),
        "base_pass_count": p0,
        "base_acceptance": _fraction(p0, n),
        "net_pass_change": repair-damage,
        "acceptance_gain": _fraction(repair-damage, n),
        "repaired": repair,
        "damaged": damage,
        "repair_rate_among_base_failures": _fraction(repair, n-p0),
        "damage_rate_among_base_successes": _fraction(damage, p0),
        "normalized_error": _quantiles(e),
        "nonfinite_normalized_error_count": int((~np.isfinite(e)).sum()),
        "exceedances": {
            str(k): {"events": int((e>k).sum()), "frequency": _fraction(int((e>k).sum()), n)}
            for k in (10, 100, 1000)
        },
    }


def evaluate_predictions(
    prediction: Any,
    reference: Any,
    baseline: Any,
    species: Sequence[str],
    *,
    initial: Any | None = None,
    reference_uncertainty: Any | None = None,
    reference_sign_stable: Any | None = None,
    atol: float = 1e-15,
    rtol: float = .1,
) -> dict[str, Any]:
    """Score aligned increment arrays without changing targets or thresholds.

    All returned acceptance scores are against supplied reference labels. An
    uncertainty screen is empirical unless the caller provides a justified bound.
    This function does not know or certify that bound, split independence, data
    provenance, state regime, or deployment validity.
    """
    if not np.isfinite([atol, rtol]).all() or atol <= 0 or not 0 <= rtol < 1:
        raise ValueError("Require finite atol>0 and 0<=rtol<1")
    truth = _array(reference, "reference")
    pred = _array(prediction, "prediction", truth.shape)
    base = _array(baseline, "baseline", truth.shape)
    names = list(species)
    if len(names) != truth.shape[1] or len(set(names)) != len(names):
        raise ValueError("Species order must be unique and match the arrays")
    if not np.isfinite(truth).all() or not np.isfinite(base).all():
        raise ValueError("Reference and frozen baseline must be finite")
    active = np.asarray([name != "AR" for name in names], dtype=bool)
    if not active.any():
        raise ValueError("No active scored species")
    t, p, b = truth[:,active], pred[:,active], base[:,active]
    tau = atol + rtol*np.abs(t)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = np.abs(p-t)/tau
        base_error = np.abs(b-t)/tau
    error = np.where(np.isfinite(error), error, np.inf)
    base_error = np.where(np.isfinite(base_error), base_error, np.inf)
    nontrivial = np.abs(t) > atol/(1-rtol)
    masks = {
        "all": np.ones(t.shape, dtype=bool),
        "nontrivial": nontrivial,
        "floor_dominated": ~nontrivial,
        "weak_nontrivial": nontrivial & (np.abs(t)<=1e-12),
        "active_increment": np.abs(t)>1e-12,
    }
    out: dict[str, Any] = {
        "scope": "Offline nominal-reference scoring; identities and custody are caller responsibilities",
        "policy": {"atol": float(atol), "rtol": float(rtol), "reference_magnitude": "increment", "comparison": "<="},
        "states": truth.shape[0],
        "scored_species": [name for name in names if name!="AR"],
        "returned_prediction_finite": bool(np.isfinite(pred).all()),
        "returned_nonfinite_components": int((~np.isfinite(pred)).sum()),
        "quantile_null_meaning": "No observations, or the corresponding empirical error quantile is nonfinite; never zero-filled",
        "groups": {name:_group(error,base_error,mask) for name,mask in masks.items()},
        "all_active_species_state_pass_count": int((error<=1).all(axis=1).sum()),
        "all_active_species_state_acceptance": float((error<=1).all(axis=1).mean()),
        "baseline_all_active_species_state_acceptance": float((base_error<=1).all(axis=1).mean()),
        "per_species": {},
        "small_target_events": [],
    }
    sign_correct = np.isfinite(p) & (p!=0) & (np.sign(p)==np.sign(t))
    out["nontrivial_sign"] = {
        "scope": "Nominal reference sign; zero predictions and nonfinite predictions are wrong on nontrivial targets",
        "components": int(nontrivial.sum()),
        "wrong_sign_count": int((nontrivial & ~sign_correct).sum()),
        "wrong_sign_rate": _fraction(int((nontrivial & ~sign_correct).sum()), int(nontrivial.sum())),
    }
    same = nontrivial & sign_correct
    with np.errstate(divide="ignore", invalid="ignore"):
        decade_error = np.abs(np.log10(np.abs(p[same]))-np.log10(np.abs(t[same])))
    out["same_sign_decade_error"] = {
        "components": int(same.sum()),
        "omitted_nontrivial_sign_or_nonfinite_failures": int((nontrivial & ~same).sum()),
        "absolute_log10_ratio": _quantiles(decade_error),
    }
    for j, name in enumerate(out["scored_species"]):
        mask = np.zeros(t.shape,dtype=bool)
        mask[:,j] = True
        out["per_species"][name] = _group(error,base_error,mask)
    for epsilon in (1e-15,1e-12,1e-9):
        selected = np.abs(t)<=epsilon
        n = int(selected.sum())
        for eta in (1e-12,1e-8,1e-6):
            if eta<=epsilon:
                continue
            k = int((selected & (np.abs(p)>=eta)).sum())
            k0 = int((selected & (np.abs(b)>=eta)).sum())
            nf = int((selected & ~np.isfinite(p)).sum())
            out["small_target_events"].append({
                "epsilon": epsilon, "eta": eta, "components": n,
                "events": k, "baseline_events": k0,
                "frequency": _fraction(k,n), "nonfinite_predictions_in_population": nf,
                "tenfold_reduction_assessable": bool(n>0 and k0>0 and nf==0),
                "conditional_error": _group(error,base_error,selected),
            })
    if initial is not None:
        start = _array(initial,"initial",truth.shape)
        if not np.isfinite(start).all():
            raise ValueError("Initial states must be finite")
        with np.errstate(over="ignore",invalid="ignore"):
            end = start[:,active]+p
            baseline_end = start[:,active]+b
        out["endpoint_diagnostics"] = {
            "negative_components": int((end<0).sum()),
            "baseline_negative_components": int((baseline_end<0).sum()),
            "nonfinite_components": int((~np.isfinite(end)).sum()),
            "scope": "Strict numerical sign counts, not a complete physical-validity certificate",
        }
    if reference_uncertainty is not None:
        uncertainty = _array(reference_uncertainty,"reference_uncertainty",truth.shape)[:,active]
        if np.any(uncertainty<0):
            raise ValueError("Uncertainty cannot be negative")
        qualified = np.isfinite(uncertainty) & (uncertainty<=.1*tau)
        out["qualified"] = {
            "scope": "Empirical uncertainty screen; not rigorous certification",
            "coverage": float(qualified.mean()),
            "all": _group(error,base_error,qualified),
            "nontrivial": _group(error,base_error,qualified & nontrivial),
        }
        if reference_sign_stable is not None:
            stable = np.asarray(reference_sign_stable)
            if stable.shape!=truth.shape or stable.dtype!=np.bool_:
                raise ValueError("reference_sign_stable must be a same-shape boolean array")
            sign_mask = qualified & nontrivial & stable[:,active] & (uncertainty<np.abs(t))
            out["qualified"]["sign"] = {
                "components": int(sign_mask.sum()),
                "wrong_sign_count": int((sign_mask & ~sign_correct).sum()),
                "wrong_sign_rate": _fraction(int((sign_mask & ~sign_correct).sum()),int(sign_mask.sum())),
            }
    elif reference_sign_stable is not None:
        raise ValueError("Sign-stability evidence requires the uncertainty array")
    argon = np.asarray([name=="AR" for name in names],dtype=bool)
    out["argon_nonzero_or_nonfinite_returned_count"] = int(((pred[:,argon]!=0) | ~np.isfinite(pred[:,argon])).sum())
    return out
