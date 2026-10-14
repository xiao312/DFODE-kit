"""Training-only scalar calibration; fixed artifact-only prediction."""
import math
import numpy as np
import torch
from ..decoder_loss.coordinates import decode


def smooth_constant(target):
    """Target [states,species]; convex constant Smooth L1 derivative root."""
    low=target.min(0).values;high=target.max(0).values
    at_zero=(-target/math.log1p(.1)).clamp(-1,1).sum(0)
    for _ in range(64):
        middle=(low+high)/2
        slope=((middle-target)/math.log1p(.1)).clamp(-1,1).sum(0)
        # If the minimizer is flat, select its endpoint nearest zero.
        move_low=(slope<0)|((slope==0)&(at_zero>0))
        low=torch.where(move_low,middle,low)
        high=torch.where(~move_low,middle,high)
    return torch.where(at_zero==0,torch.zeros_like(low),(low+high)/2)


def interval_midpoint(low,high):
    """Maximum interior overlap; stable ties, no unbounded endpoint tricks."""
    if low.shape!=high.shape or not torch.isfinite(low).all() or not torch.isfinite(high).all() or (low>=high).any():
        raise ValueError("Invalid acceptable intervals")
    values=torch.cat([low,high],dim=0).T
    events=torch.cat([torch.ones_like(low,dtype=torch.int64),-torch.ones_like(high,dtype=torch.int64)],dim=0).T
    endpoints,order=torch.sort(values,dim=1,stable=True)
    coverage=torch.gather(events,1,order).cumsum(1)[:,:-1]
    left,right=endpoints[:,:-1],endpoints[:,1:]
    midpoint=left+(right-left)/2
    valid=(right>left)&(midpoint>left)&(midpoint<right)
    counts=torch.where(valid,coverage,-1)
    best=counts.max(1).values
    candidates=torch.where(valid&(counts==best[:,None]),midpoint.abs(),torch.inf)
    ix=candidates.argmin(1)
    chosen=midpoint[torch.arange(len(ix),device=ix.device),ix]
    chosen=torch.where(best>=0,chosen,torch.zeros_like(chosen))
    return chosen,best


def calibrated_prediction(base,constants,active):
    c=np.broadcast_to(constants,(len(base),len(constants)))
    output=base.copy();output[:,active]=decode(base[:,active],c,dict(decoder="asinh",scale_floor=1e-14))
    return output,c


def train_constants(rows,arm,device="cuda"):
    d=rows["delta"][:,rows["active"]];b=rows["base"][:,rows["active"]]
    if arm=="smooth_l1":
        constants=smooth_constant(torch.as_tensor(rows["target"],device=device,dtype=torch.float64)).cpu().numpy()
        detail=dict(bisection_steps=64)
    elif arm=="interval":
        truth=torch.as_tensor(d,device=device,dtype=torch.float64)
        baseline=torch.as_tensor(b,device=device,dtype=torch.float64)
        allowance=1e-15+.1*truth.abs();origin=torch.asinh(baseline/1e-14)
        low=torch.asinh((truth-allowance)/1e-14)-origin
        high=torch.asinh((truth+allowance)/1e-14)-origin
        chosen,covered=interval_midpoint(low,high)
        constants=chosen.cpu().numpy()
        proposed=decode(b,np.broadcast_to(constants,b.shape),dict(decoder="asinh",scale_floor=1e-14))
        count=(abs(proposed-d)<=1e-15+.1*abs(d)).sum(0)
        base_count=(abs(b-d)<=1e-15+.1*abs(d)).sum(0)
        fallback=count<=base_count;constants[fallback]=0
        detail=dict(interior_max_counts=covered.cpu().tolist(),proposed_physical_counts=count.tolist(),
                    base_physical_counts=base_count.tolist(),zero_preferred_or_fallback=fallback.tolist())
    else:raise ValueError("Unknown calibration arm")
    pred,c=calibrated_prediction(rows["base"],constants,rows["active"])
    return constants,pred,c,detail
