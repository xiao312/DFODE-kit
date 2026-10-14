"""Physical repair/preservation and monotone-coordinate interval diagnostics."""
import numpy as np
from ..decoder_loss.coordinates import decode
from ..decoder_loss.diagnostics import quantiles
from ..correction_scaling.tail_metrics import physical_scores


def score(prediction,truth,base):
    result = physical_scores(prediction,truth,base)
    passed = abs(base-truth)<=1e-15+.1*abs(truth)
    successes,failures = int(passed.sum()),int((~passed).sum())
    result.update(base_successes=successes,base_failures=failures,
        damage_rate=result["pass_to_fail"]/successes if successes else None,
        repair_rate=result["fail_to_pass"]/failures if failures else None,
        net_pass_change=result["fail_to_pass"]-result["pass_to_fail"])
    return result


def preservation(base,truth,coordinate,config):
    allowance = 1e-15+.1*abs(truth)
    old_error = abs(base-truth)/allowance
    s = config["scale_floor"]
    origin = np.arcsinh(base/s)
    lower = np.arcsinh((truth-allowance)/s)-origin
    upper = np.arcsinh((truth+allowance)/s)-origin
    target = np.arcsinh(truth/s)-origin
    prediction = decode(base,coordinate,config)
    physical_pass = abs(prediction-truth)<=allowance
    interval_pass = (coordinate>=lower)&(coordinate<=upper)
    disagreement = physical_pass!=interval_pass
    distance = np.maximum.reduce([lower-coordinate,coordinate-upper,np.zeros_like(coordinate)])
    boundary_distance = np.minimum(abs(coordinate-lower),abs(coordinate-upper))
    if np.any(disagreement & (boundary_distance>1e-11)):
        raise ValueError("Interval and physical decoder disagree beyond roundoff boundary")
    records = []
    for lo,hi in ((0.,.1),(.1,.5),(.5,.8),(.8,1.)):
        mask = (old_error>=lo if lo==0 else old_error>lo)&(old_error<=hi)
        damaged = mask&~physical_pass
        away = damaged&(coordinate*target<0)
        overshoot = damaged&(coordinate*target>0)
        center_zero = damaged&(coordinate*target==0)
        records.append(dict(lower=lo,upper=hi,base_successes=int(mask.sum()),damaged=int(damaged.sum()),
            damage_rate=float(damaged.sum()/mask.sum()) if mask.any() else None,
            away_from_reference_center=int(away.sum()),same_direction_overshoot=int(overshoot.sum()),
            zero_center_or_correction=int(center_zero.sum()),
            coordinate_violation_in_beta_units=quantiles(distance[damaged]/np.log1p(.1)),
            corrected_normalized_error=quantiles((abs(prediction-truth)/allowance)[damaged])))
    return dict(bins=records,interval_disagreements=int(disagreement.sum()),
        maximum_disagreement_boundary_distance=float(boundary_distance[disagreement].max()) if disagreement.any() else 0.)
