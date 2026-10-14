"""Increment-only tails and oracle tests, with explicit populations and empty bins."""
import numpy as np
from benchmarks.offline_accuracy.metrics import MAGNITUDE_EDGES
from .coordinates import encode,decode


def quantiles(value):
    return (dict(zip(("min","p50","p95","p99","max"),map(float,np.quantile(value,[0,.5,.95,.99,1]))))
            if value.size else {key:None for key in ("min","p50","p95","p99","max")})


def population(predicted,truth,mask):
    error = abs(predicted-truth)
    ratio = error/(1e-15+.1*abs(truth))
    count = int(mask.sum())
    passing = int(((ratio<=1)&mask).sum())
    return dict(components=count,pass_count=passing,acceptance=passing/count if count else None,
                normalized_error=quantiles(ratio[mask]))


def tails(predicted,truth,species,uncertainty=None):
    active = np.array([name!="AR" for name in species])
    p,d = predicted[:,active],truth[:,active]
    nontrivial = abs(d)>1e-15/.9
    selected = np.ones_like(d,dtype=bool)
    result = dict(all=population(p,d,selected),nontrivial=population(p,d,nontrivial),
        sign=dict(components=int(nontrivial.sum()),wrong_sign_count=int(((np.sign(p)!=np.sign(d))&nontrivial).sum())),
        overprediction=[],species=[],magnitude_bins=[])
    for epsilon in (1e-15,1e-12,1e-9):
        small = abs(d)<=epsilon
        for eta in (1e-12,1e-8,1e-6):
            if eta<=epsilon:
                continue
            n,k = int(small.sum()),int((small&(abs(p)>=eta)).sum())
            result["overprediction"].append(dict(epsilon=epsilon,eta=eta,events=k,components=n,
                frequency=k/n if n else None,conditional=population(p,d,small)))
    for j,name in enumerate(np.array(species)[active]):
        result["species"].append(dict(species=name,**population(p[:,j],d[:,j],selected[:,j])))
    for lo,hi in zip(MAGNITUDE_EDGES[:-1],MAGNITUDE_EDGES[1:]):
        result["magnitude_bins"].append(dict(lower=lo,upper=None if np.isinf(hi) else hi,
            **population(p,d,(abs(d)>=lo)&(abs(d)<hi))))
    if uncertainty is not None:
        u = uncertainty[:,active]
        qualified = np.isfinite(u)&(u<=.1*(1e-15+.1*abs(d)))&nontrivial
        result["resolved_nontrivial"] = population(p,d,qualified)
        result["resolved_nontrivial"]["scope"] = "Existing audited subset only; empirical uncertainty, not a rigorous bound"
    return result


def oracle(base,truth,species,config,cuda=False):
    active = np.array([name!="AR" for name in species])
    b,d = base[:,active],truth[:,active]
    s,tau = config["scale_floor"],1e-15+.1*abs(d)
    c = encode(b,d,config)
    single = c.astype(np.float32)
    full = decode(b,c,config)
    nearest = decode(b,single,config)
    best = abs(nearest-d)/tau
    for direction in (-np.inf,np.inf):
        neighbor = np.nextafter(single,np.float32(direction))
        best = np.minimum(best,abs(decode(b,neighbor,config)-d)/tau)
    if config["decoder"] == "linear":
        derivative = s+abs(b)
        width = 2*tau/derivative
    else:
        derivative = np.hypot(s,d)
        width = np.arcsinh((d+tau)/s)-np.arcsinh((d-tau)/s)
    selected = np.ones_like(d,dtype=bool)
    result = dict(decoder=config["decoder"],coordinate_normalization=dict(offset=0.,divisor=1.),
        fp64=population(full,d,selected),fp32=population(nearest,d,selected),
        fp32_neighbor_best=dict(components=int(d.size),pass_count=int((best<=1).sum()),normalized_error=quantiles(best)),
        coordinate=quantiles(c),local_allowed_coordinate_error=quantiles(tau/derivative),
        accepted_coordinate_interval_width=quantiles(width),species=[],magnitude_bins=[])
    if cuda:
        import torch
        from .coordinates import torch_decode
        for name,coordinates in (("cuda_fp32",single),("cuda_fp64",c)):
            pieces = []
            with torch.no_grad():
                for start in range(0,len(b),4096):
                    pieces.append(torch_decode(torch.as_tensor(b[start:start+4096],device="cuda"),
                        torch.as_tensor(coordinates[start:start+4096],device="cuda"),config).cpu().numpy())
            result[name] = population(np.concatenate(pieces),d,selected)
    for j,name in enumerate(np.array(species)[active]):
        result["species"].append(dict(species=name,**population(nearest[:,j],d[:,j],selected[:,j])))
    for lo,hi in zip(MAGNITUDE_EDGES[:-1],MAGNITUDE_EDGES[1:]):
        mask = (abs(d)>=lo)&(abs(d)<hi)
        result["magnitude_bins"].append(dict(lower=lo,upper=None if np.isinf(hi) else hi,
            **population(nearest,d,mask),allowed_coordinate_error=quantiles((tau/derivative)[mask])))
    return result
