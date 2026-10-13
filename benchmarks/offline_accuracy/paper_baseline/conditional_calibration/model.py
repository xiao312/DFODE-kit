"""Fixed-bin scalar maps fitted on training labels, queried without labels."""
import time
import numpy as np
from ..interval_calibration.model import train_constants
from ..decoder_loss.coordinates import decode


def partition(base,bins):
    """Interior fit-only quantiles; repeated values cannot make empty intervals."""
    z=np.arcsinh(np.asarray(base,dtype=float)/1e-14)
    edges=np.unique(np.quantile(z,np.arange(1,bins)/bins)) if bins>1 else np.empty(0)
    return edges,np.searchsorted(edges,z,side="right")


def fit_map(base,truth,bins,minimum,device="cuda",deadline=None):
    if base.shape!=truth.shape or base.ndim!=2:raise ValueError("Species alignment differs")
    models=[]
    for i in range(base.shape[1]):
        edges,groups=partition(base[:,i],bins);constants=[];counts=[];details=[]
        for group in range(len(edges)+1):
            if deadline is not None and time.monotonic()>deadline:raise TimeoutError("Registered finite worker budget exceeded")
            selected=groups==group;count=int(selected.sum());counts.append(count)
            if count<minimum:
                constants.append(0.);details.append(dict(sparse_identity=True));continue
            b=base[selected,i:i+1];d=truth[selected,i:i+1]
            rows=dict(base=b,delta=d,active=np.ones(1,dtype=bool),target=np.arcsinh(d/1e-14)-np.arcsinh(b/1e-14))
            c,_,_,detail=train_constants(rows,"interval",device=device)
            constants.append(float(c[0]));details.append(detail)
        models.append(dict(edges=edges.tolist(),constants=constants,counts=counts,details=details))
    return models


def predict(base,models):
    if base.ndim!=2 or base.shape[1]!=len(models):raise ValueError("Species axis differs")
    correction=np.empty_like(base,dtype=float)
    for i,model in enumerate(models):
        edges=np.asarray(model["edges"]);constants=np.asarray(model["constants"])
        if len(constants)!=len(edges)+1 or np.any(np.diff(edges)<=0):raise ValueError("Invalid scalar artifact")
        correction[:,i]=constants[np.searchsorted(edges,np.arcsinh(base[:,i]/1e-14),side="right")]
    return decode(base,correction,dict(decoder="asinh",scale_floor=1e-14))


def fold_ids(count,seed):
    ids=np.empty(count,dtype=np.int64)
    ids[np.random.default_rng(seed+411).permutation(count)]=np.arange(count)%2
    return ids
