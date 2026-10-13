"""Small comparative classifier and fixed, label-free replacement rule."""
import numpy as np
import torch
import math


class Gate(torch.nn.Module):
    def __init__(self,features,species,arm):
        super().__init__()
        if arm not in ("intercept","affine"):raise ValueError("Unknown gate")
        self.bias=torch.nn.Parameter(torch.zeros(species))
        self.weight=torch.nn.Parameter(torch.zeros(features,species),requires_grad=arm=="affine")
        self.own=torch.nn.Parameter(torch.zeros(2,species),requires_grad=arm=="affine")

    def forward(self,x,c):
        return x@self.weight+self.bias+c*self.own[0]+c.abs()*self.own[1]


class NonlinearGate(torch.nn.Module):
    """Same per-species inputs as affine; only the function class changes."""
    def __init__(self,features,species,seed,width=16):
        super().__init__();torch.manual_seed(seed)
        bound=1/math.sqrt(features+2)
        self.hidden_weight=torch.nn.Parameter(torch.empty(species,features,width).uniform_(-bound,bound))
        self.hidden_own=torch.nn.Parameter(torch.empty(species,2,width).uniform_(-bound,bound))
        self.hidden_bias=torch.nn.Parameter(torch.empty(species,width).uniform_(-bound,bound))
        self.output_weight=torch.nn.Parameter(torch.zeros(species,width))
        self.bias=torch.nn.Parameter(torch.zeros(species))

    def forward(self,x,c):
        h=torch.einsum("bf,sfh->bsh",x,self.hidden_weight)+self.hidden_bias
        h=h+c[:,:,None]*self.hidden_own[:,0]+c.abs()[:,:,None]*self.hidden_own[:,1]
        return (torch.nn.functional.gelu(h)*self.output_weight).sum(-1)+self.bias


def make_gate(features,species,arm,seed):
    return NonlinearGate(features,species,seed) if arm=="nonlinear" else Gate(features,species,arm)


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
