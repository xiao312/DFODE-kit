"""Frozen geometry and learned coefficients; no reference lookup at inference."""
import numpy as np
import torch
from ..adaptive.model import features,base_preprocessing
from ..correction_scaling.model import base_prediction,fixed_batches
from ..decoder_loss.coordinates import decode


def select_centres(ids,snapshots,seed,count=128):
    unique = np.unique(snapshots)
    if count % len(unique) or len(np.unique(ids)) != len(ids):
        raise ValueError("Invalid centre count or IDs")
    rng = np.random.default_rng(seed)
    chosen = []
    for snapshot in unique:
        ix = np.flatnonzero(snapshots==snapshot)
        ix = ix[np.argsort(ids[ix],kind="stable")]
        chosen.extend(np.sort(rng.choice(ix,count//len(unique),replace=False)))
    return np.array(chosen)


class Basis(torch.nn.Module):
    def __init__(self,centres,weights,bandwidth):
        super().__init__()
        if (weights.ndim != 2 or centres.shape[1] != weights.shape[1]
                or not np.isfinite(weights).all() or np.any(weights<=0)
                or not np.isfinite(centres).all() or np.any(bandwidth<=0)
                or not np.isfinite(bandwidth).all()):
            raise ValueError("Invalid frozen geometry")
        self.register_buffer("centres",torch.as_tensor(centres,dtype=torch.float64))
        self.register_buffer("weights",torch.as_tensor(weights,dtype=torch.float64))
        self.register_buffer("bandwidth",torch.as_tensor(bandwidth,dtype=torch.float64))
        self.register_buffer("cross",(self.weights[:,None,:]*self.centres[None,:,:]).reshape(-1,centres.shape[1]).T.contiguous())
        self.register_buffer("cnorm",self.weights @ self.centres.square().T)

    def distances(self,x):
        x = x.double()
        squared = (x.square() @ self.weights.T)[:,:,None]+self.cnorm[None,:,:]
        squared -= 2*(x @ self.cross).reshape(len(x),len(self.weights),len(self.centres))
        return squared.clamp_min(0)

    def forward(self,x):
        kernel = torch.exp(-self.distances(x)/(2*self.bandwidth[None,:,None]))
        return (kernel/(1+kernel.sum(dim=2,keepdim=True))).float()


def geometry(x,centres,weights,device="cuda"):
    sample = np.setdiff1d(np.arange(len(x)),centres)[:4096]
    basis = Basis(x[centres],weights,np.ones(len(weights))).to(device)
    with torch.no_grad():
        distances = basis.distances(torch.as_tensor(x[sample],device=device)).min(dim=2).values.cpu().numpy()
    bandwidth,fallback = [],[]
    for values in distances.T:
        value = float(np.median(values))
        used = value <= 0
        if used:
            positive = values[values>0]
            value = float(np.median(positive)) if len(positive) else 1.
        bandwidth.append(value);fallback.append(used)
    return np.array(bandwidth),fallback


def cached_coordinate(basis,coefficients,x):
    values=[]
    with torch.no_grad():
        for count,chunk in fixed_batches(x):
            phi=basis(torch.as_tensor(chunk,device=coefficients.device))
            values.append(torch.einsum("nsk,sk->ns",phi,coefficients).double().cpu().numpy()[:count])
    return np.concatenate(values)


def predict(base,prep,basis,coefficients,states,seed):
    b,_=base_prediction(base,base_preprocessing(prep),states,seed)
    x=features(states,b,prep,dict(scale_floor=1e-14)).astype(np.float32)
    c=cached_coordinate(basis,coefficients,x)
    output=b.copy()
    output[:,prep["active"]]=decode(b[:,prep["active"]],c,dict(decoder="asinh",scale_floor=1e-14))
    return output,c
