"""Physical scoring, fixed group reports and explicit coordinate diagnostics."""
import numpy as np
from ..decoder_loss.diagnostics import tails,quantiles
from ..decoder_loss.coordinates import encode
from .coverage import group_scores


def summary(predicted,coordinate,rows,base,species,config):
    active = np.array([s!="AR" for s in species])
    ref = rows["delta"]
    tau = 1e-15+.1*abs(ref[:,active])
    passed = abs(predicted[:,active]-ref[:,active])<=tau
    old = abs(base[:,active]-ref[:,active])<=tau
    target = encode(base[:,active],ref[:,active],config)
    error = abs(coordinate-target)
    sufficient = error <= np.log1p(.1)
    if (sufficient & ~passed).any():
        raise ValueError("Coordinate sufficient-bound check failed in actual decoder")
    result = tails(predicted,ref,species)
    # Independently count direct physical decisions, not transformed-space decisions.
    if result["all"]["pass_count"] != int(passed.sum()):
        raise ValueError("Physical acceptance count disagreement")
    result.update(whole_states_passed=int(passed.all(axis=1).sum()),states=len(ref),
        coordinate_error=quantiles(error),coordinate_sufficient_pass_count=int(sufficient.sum()),
        pass_to_fail=int((old&~passed).sum()),fail_to_pass=int((~old&passed).sum()),
        negative_endpoint_components=int(((rows["states"][:,2:]+predicted)[:,active]<0).sum()),
        groups=group_scores(predicted,rows,species))
    return result


def audit_scores(predicted,rows,audit,species,split):
    lookup = {int(v):i for i,v in enumerate(rows["source_indices"])}
    records = [r for r in audit["records"] if r["split"]==split and r["source_row"] in lookup]
    if not records:
        return dict(states=0,scope="No audited rows in this correction-training subset")
    indices = [lookup[r["source_row"]] for r in records]
    reference = np.array([r["reference_delta"] for r in records])
    uncertainty = np.array([r["uncertainty_estimate"] for r in records])
    start = rows["states"][indices,2:]
    uncertainty = np.maximum.reduce([uncertainty,abs(np.spacing(start)),abs(np.spacing(start+reference))])
    return dict(states=len(indices),metrics=tails(predicted[indices],reference,species,uncertainty))
