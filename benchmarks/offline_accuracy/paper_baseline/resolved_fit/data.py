"""Training-only sample selection and scores with explicit resolution masks."""
import numpy as np
from ..decoder_loss.diagnostics import population


def selected_rows(training, audit, active, expected_rows=16):
    records = sorted((r for r in audit["records"] if r["split"] == "train"),
                     key=lambda r:r["source_row"])
    ids = [r["source_row"] for r in records]
    if len(ids) != expected_rows or len(set(ids)) != expected_rows:
        raise ValueError("Require the fixed distinct training audit rows")
    lookup = {int(v):i for i,v in enumerate(training["source_indices"])}
    if any(v not in lookup for v in ids) or any(r["status"] != "checked" for r in records):
        raise ValueError("Audit row absent from training or not checked")
    indices = [lookup[v] for v in ids]
    states = training["states"][indices]
    reference = np.asarray([r["reference_delta"] for r in records], dtype=float)
    uncertainty = np.asarray([r["uncertainty_estimate"] for r in records], dtype=float)
    if reference.shape != states[:,2:].shape or uncertainty.shape != reference.shape:
        raise ValueError("Reference shape differs from state")
    if not np.isfinite(reference).all() or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
        raise ValueError("Invalid audited reference or uncertainty")
    uncertainty = np.maximum.reduce([uncertainty, abs(np.spacing(states[:,2:])),
                                    abs(np.spacing(states[:,2:]+reference))])
    resolved = uncertainty <= .1*(1e-15+.1*abs(reference))
    resolved &= np.asarray(active)[None,:]
    if np.any(resolved.sum(axis=1)==0):
        raise ValueError("An audited state has no resolved active component")
    return dict(states=states, reference=reference, uncertainty=uncertainty,
                resolved=resolved, source_indices=np.asarray(ids),
                ordinary_delta=training["delta"][indices])


def scores(prediction, reference, resolved, base, species):
    nontrivial = resolved & (abs(reference)>1e-15/.9)
    passing = abs(prediction-reference) <= 1e-15+.1*abs(reference)
    base_failed = resolved & (abs(base-reference)>1e-15+.1*abs(reference))
    active = np.asarray([name != "AR" for name in species])
    whole_resolved = resolved[:,active].all(axis=1)
    subset_pass = (passing | ~resolved).all(axis=1)
    return dict(resolved=population(prediction,reference,resolved),
        nontrivial=population(prediction,reference,nontrivial),
        originally_failed=population(prediction,reference,base_failed),
        sign_errors=int(((np.sign(prediction)!=np.sign(reference)) & nontrivial).sum()),
        resolved_subset_rows_passed=int(subset_pass.sum()),
        fully_resolved_rows=int(whole_resolved.sum()),
        whole_state_pass_count=int((subset_pass & whole_resolved).sum()),
        species=[dict(species=name,**population(prediction[:,j],reference[:,j],resolved[:,j]))
                 for j,name in enumerate(species) if name != "AR"])
