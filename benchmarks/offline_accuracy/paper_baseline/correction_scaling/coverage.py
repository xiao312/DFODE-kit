"""Read-only component coverage and reference comparison, with explicit empty bins."""
import numpy as np
from ..decoder_loss.diagnostics import population,quantiles

UPPER = np.array([1e-15,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,np.inf])
LABELS = ["zero","(0,1e-15]","(1e-15,1e-14]","(1e-14,1e-12]",
          "(1e-12,1e-10]","(1e-10,1e-8]","(1e-8,1e-6]","(1e-6,1e-4]","(1e-4,inf)"]


def magnitude_bins(delta):
    if not np.isfinite(delta).all():
        raise ValueError("Finite increments required")
    return np.where(delta==0,0,1+np.searchsorted(UPPER,abs(delta),side="left"))


def ranges(values):
    return dict(min=float(values.min()),max=float(values.max())) if len(values) else dict(min=None,max=None)


def profile(rows,species):
    states,delta = rows["states"],rows["delta"]
    if len(np.unique(rows["source_indices"])) != len(states):
        raise ValueError("Duplicate split-local source row identifiers")
    if not np.isfinite(states).all() or not np.isfinite(delta).all():
        raise ValueError("Nonfinite source data")
    groups = magnitude_bins(delta)
    result = dict(states=len(states),unique_states=len(np.unique(states,axis=0)),
        temperature_K=ranges(states[:,0]),pressure_Pa=ranges(states[:,1]),
        sources={str(s):int((rows["snapshot"]==s).sum()) for s in np.unique(rows["snapshot"])},groups=[])
    fuel = [species.index(s) for s in ("NH3","CH4") if s in species]
    fuel_fraction = states[:,2:][:,fuel].sum(axis=1)
    for j,name in enumerate(species):
        if name == "AR":
            continue
        for index,label in enumerate(LABELS):
            mask = groups[:,j]==index
            n = int(mask.sum())
            result["groups"].append(dict(species=name,bin=label,components=n,distinct_states=n,
                distinct_sources=len(np.unique(rows["snapshot"][mask])),
                positive=int((delta[mask,j]>0).sum()),negative=int((delta[mask,j]<0).sum()),
                zero=int((delta[mask,j]==0).sum()),temperature_K=ranges(states[mask,0]),
                pressure_Pa=ranges(states[mask,1]),fuel_mass_fraction=ranges(fuel_fraction[mask]),
                reference_quality="Nominal labels; independent checks available only on audited subset"))
    active = np.array([s!="AR" for s in species])
    result["magnitude_totals"] = [dict(bin=label,components=int((groups[:,active]==index).sum()),
        distinct_states=int((groups[:,active]==index).any(axis=1).sum())) for index,label in enumerate(LABELS)]
    return result


def support(training,development):
    by_key = {(g["species"],g["bin"]):g for g in training["groups"]}
    return [dict(species=d["species"],bin=d["bin"],development_components=d["components"],
        training_components=by_key[(d["species"],d["bin"])]["components"],
        status="absent" if by_key[(d["species"],d["bin"])]["components"]==0 else "sparse")
        for d in development["groups"] if d["components"]>0 and by_key[(d["species"],d["bin"])]["components"]<128]


def label_check(ordinary,reference,uncertainty,active):
    ratio = abs(ordinary-reference)/(1e-15+.1*abs(reference))
    mask = (uncertainty<=.1*(1e-15+.1*abs(reference))) & active[None,:]
    return dict(checked_components=int(mask.sum()),ratio=quantiles(ratio[mask]),
        over_tenth_allowance=int(((ratio>.1)&mask).sum()),over_full_allowance=int(((ratio>1)&mask).sum()),
        passed=bool(mask.any() and not ((ratio>.1)&mask).any()),
        scope="Existing audited states only; no certification of unaudited labels")


def group_scores(predicted,rows,species):
    delta = rows["delta"]
    groups = magnitude_bins(delta)
    result = []
    for j,name in enumerate(species):
        if name=="AR":
            continue
        for index,label in enumerate(LABELS):
            result.append(dict(species=name,bin=label,**population(predicted[:,j],delta[:,j],groups[:,j]==index)))
    active = np.array([s!="AR" for s in species])
    temperatures = []
    for lo,hi in zip([0,305,500,1000,1500,2000],[305,500,1000,1500,2000,np.inf]):
        mask = ((rows["states"][:,0]>=lo)&(rows["states"][:,0]<hi))[:,None]&active[None,:]
        temperatures.append(dict(lower_K=lo,upper_K=float(hi) if np.isfinite(hi) else None,
                                  **population(predicted,delta,mask)))
    return dict(species_magnitude=result,temperature=temperatures)
