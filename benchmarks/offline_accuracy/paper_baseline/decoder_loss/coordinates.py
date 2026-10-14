"""Label-free FP64 decoders and exact oracle coordinates for diagnostic use only."""
import numpy as np


def encode(base,truth,config):
    s = config["scale_floor"]
    if config["decoder"] == "linear":
        return (truth-base)/(s+abs(base))
    if config["decoder"] == "asinh":
        return np.arcsinh(truth/s)-np.arcsinh(base/s)
    raise ValueError("Unknown decoder")


def decode(base,coordinate,config):
    base,c = np.broadcast_arrays(np.asarray(base,dtype=float),np.asarray(coordinate,dtype=float))
    s = config["scale_floor"]
    if config["decoder"] == "linear":
        output = base+(s+abs(base))*c
    elif config["decoder"] == "asinh":
        near = abs(c)<=.5
        output = np.empty_like(base)
        b,q = base[near],c[near]
        output[near] = b+(2*b*np.sinh(q/2)**2+np.hypot(b,s)*np.sinh(q))
        output[~near] = s*np.sinh(np.arcsinh(base[~near]/s)+c[~near])
    else:
        raise ValueError("Unknown decoder")
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite decoder; no clipping fallback")
    return output


def torch_decode(base,coordinate,config):
    import torch
    b,c = base.double(),coordinate.double()
    s = config["scale_floor"]
    if config["decoder"] == "linear":
        return b+(s+b.abs())*c
    if config["decoder"] != "asinh":
        raise ValueError("Unknown decoder")
    near = c.abs()<=.5
    out = torch.empty_like(c)
    # Index branches, not where(c==0, b, ...): keep zero-init gradients alive.
    out[near] = b[near]+(2*b[near]*torch.sinh(c[near]/2).square()
                        +torch.hypot(b[near],torch.full_like(b[near],s))*torch.sinh(c[near]))
    out[~near] = s*torch.sinh(torch.asinh(b[~near]/s)+c[~near])
    return out


def physical_loss(predicted,truth,config):
    import torch
    u = (predicted-truth)/(config["atol"]+config["rtol"]*truth.abs())
    if config["objective"] == "log":
        return torch.log1p(u.abs()).mean()
    if config["objective"] == "huber":
        return torch.nn.functional.huber_loss(u,torch.zeros_like(u),delta=1.,reduction="mean")
    raise ValueError("Unknown physical loss")
