"""Shared input work and FP64 device reconstruction; frozen networks unchanged."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from ..adaptive.model import base_preprocessing
from ..correction_scaling.model import fixed_batches
from .metrics import bounded_mean, strict_shrink


def inverse(initial, coordinate):
    """Exact historical branch definitions; dummy positive arguments avoid NaNs.

    Negative predicted power bases map to zero endpoints as in the historical
    GBCT decoder. Expose the mask, never apply another endpoint repair.
    """
    encoded=1e-6*.25*coordinate*coordinate.abs()
    b0=initial.pow(.1);b1=b0+.1*encoded
    ratio=.1*encoded/torch.where(b0>0,b0,torch.ones_like(b0))
    moderate=ratio.abs()<.5
    logsmall=torch.log1p(torch.where(moderate,ratio,torch.zeros_like(ratio)))
    loglarge=torch.log(torch.where(b1>0,b1,torch.ones_like(b1)))-torch.log(torch.where(b0>0,b0,torch.ones_like(b0)))
    value=initial*torch.expm1(torch.where(moderate,logsmall,loglarge)/.1)
    value=torch.where(b1<=0,-initial,value)
    value=torch.where(initial==0,b1.clamp_min(0).pow(10.),value)
    return value,b1<0


def predict_bases(models, states):
    """No references; both sequential frozen forwards before the only D2H copy."""
    seeds=list(models);preps={s:base_preprocessing(models[s][1]) for s in seeds}
    active=preps[seeds[0]]["active"]
    for s in seeds:np.testing.assert_array_equal(preps[s]["active"],active)
    values={s:[] for s in seeds};masks={s:[] for s in seeds}
    for count,chunk in fixed_batches(states):
        features=input_features(chunk);coords=[]
        with torch.no_grad():
            for s in seeds:
                model,_=models[s];p=preps[s]
                x=(features-p["x_offset"])/p["x_scale"]
                z=model(torch.as_tensor(x,dtype=torch.float32,device="cuda")).double()
                coords.append(z*torch.as_tensor(p["y_scale"],device="cuda")+torch.as_tensor(p["y_offset"],device="cuda"))
            initial=torch.as_tensor(chunk[:,2:][:,active],dtype=torch.float64,device="cuda")
            y,mask=inverse(initial[None],torch.stack(coords))
            packed=torch.stack((y,mask.double())).cpu().numpy()
        if not np.isfinite(packed).all():raise ValueError("Nonfinite GPU inverse")
        for i,s in enumerate(seeds):
            full=np.zeros_like(chunk[:,2:]);full[:,active]=packed[0,i]
            fullmask=np.zeros_like(full,dtype=bool);fullmask[:,active]=packed[1,i].astype(bool)
            values[s].append(full[:count]);masks[s].append(fullmask[:count])
    return {s:np.concatenate(values[s]) for s in seeds},{s:np.concatenate(masks[s]) for s in seeds}


def predict(models,states,seed,fraction,mode):
    if mode!="strict-shrink":raise ValueError("GPU trial is strict-shrink only")
    values,masks=predict_bases(models,states);other=next(s for s in models if s!=seed)
    b,q=values[seed],values[other]
    return strict_shrink(b,bounded_mean(b,q,fraction),fraction),b,q,masks
