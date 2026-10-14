"""Small independent helpers for gradient geometry and finite-step counts."""
import numpy as np
import torch


def geometry(gradients):
    values=torch.stack(gradients).double()
    dots=values@values.T;norms=values.norm(dim=1)
    denominator=norms[:,None]*norms[None,:]
    result=[]
    for i in range(len(values)):
        result.append([float((dots[i,j]/denominator[i,j]).clamp(-1,1)) if denominator[i,j]>0 else None for j in range(len(values))])
    return dict(norms=norms.cpu().tolist(),cosines=result,dots=dots.cpu().tolist())


def physical_counts(prediction,truth,baseline,masks):
    budget=1e-15+.1*abs(truth);old=abs(baseline-truth)<=budget;passed=abs(prediction-truth)<=budget
    loss=np.log1p(abs(prediction-truth)/budget)
    return {name:dict(components=int(mask.sum()),passes=int((passed&mask).sum()),
                     repaired=int((passed&~old&mask).sum()),damaged=int((~passed&old&mask).sum()),
                     loss=float(loss[mask].mean()) if mask.any() else None) for name,mask in masks.items()}


def groups(truth,baseline):
    passed=abs(baseline-truth)<=1e-15+.1*abs(truth);nt=abs(truth)>1e-15/.9
    return dict(all=np.ones_like(passed),base_pass=passed,base_fail=~passed,
                floor=~nt,weak_nontrivial=nt&(abs(truth)<=1e-12),active_increment=abs(truth)>1e-12)
