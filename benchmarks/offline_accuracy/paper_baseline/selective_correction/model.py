"""Small comparative classifier and fixed, label-free replacement rule."""
import numpy as np
import torch


class Gate(torch.nn.Module):
    def __init__(self,features,species,arm):
        super().__init__()
        if arm not in ("intercept","affine"):raise ValueError("Unknown gate")
        self.bias=torch.nn.Parameter(torch.zeros(species))
        self.weight=torch.nn.Parameter(torch.zeros(features,species),requires_grad=arm=="affine")
        self.own=torch.nn.Parameter(torch.zeros(2,species),requires_grad=arm=="affine")

    def forward(self,x,c):
        return x@self.weight+self.bias+c*self.own[0]+c.abs()*self.own[1]


def replace(base,candidate,initial,probability,threshold):
    """No reference input; exact branch and declared nonnegative-endpoint guard."""
    allowed=(initial+candidate)>=0
    selected=(probability>=threshold)&allowed
    return np.where(selected,candidate,base),selected


def calibrate(probability,base,candidate,initial,truth):
    """Fixed threshold grid; empirical preservation, never a risk certificate."""
    allowance=1e-15+.1*abs(truth);old=abs(base-truth)<=allowance;new=abs(candidate-truth)<=allowance
    nt=abs(truth)>1e-15/.9;oldsign=(np.sign(base)!=np.sign(truth))&nt;newsign=(np.sign(candidate)!=np.sign(truth))&nt
    best=(0,0,1.01);table=[]
    for threshold in np.r_[np.linspace(0,1,101),1.01]:
        _,selected=replace(base,candidate,initial,probability,float(threshold))
        passed=np.where(selected,new,old);sign=np.where(selected,newsign,oldsign)
        damage=int((selected&old&~new).sum());gain=int(passed.sum()-old.sum());count=int(selected.sum())
        valid=(damage<=.01*old.sum() and (passed&nt).sum()>=(old&nt).sum() and sign.sum()<=oldsign.sum())
        table.append(dict(threshold=float(threshold),gain=gain,damage=damage,selected=count,valid=bool(valid)))
        rank=(gain,-count,float(threshold))
        if valid and rank>best:best=rank
    return best[2],table
