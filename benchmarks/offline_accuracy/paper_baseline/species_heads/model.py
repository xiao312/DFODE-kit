"""Independent species heads evaluated together; no inference routing."""
import math
import torch
from ..resolved_fit.fit import head


class SpeciesHeads(torch.nn.Module):
    def __init__(self, inputs, outputs, seed):
        super().__init__()
        # Local RNG context prevents construction from changing batch RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.w1=torch.nn.Parameter(torch.empty(outputs,inputs,32).uniform_(-1/math.sqrt(inputs),1/math.sqrt(inputs)))
            self.b1=torch.nn.Parameter(torch.empty(outputs,32).uniform_(-1/math.sqrt(inputs),1/math.sqrt(inputs)))
            self.w2=torch.nn.Parameter(torch.empty(outputs,32,32).uniform_(-1/math.sqrt(32),1/math.sqrt(32)))
            self.b2=torch.nn.Parameter(torch.empty(outputs,32).uniform_(-1/math.sqrt(32),1/math.sqrt(32)))
            self.w3=torch.nn.Parameter(torch.zeros(outputs,32))
            self.b3=torch.nn.Parameter(torch.zeros(outputs))

    def forward(self,x):
        h=torch.nn.functional.gelu(torch.einsum("bi,sih->bsh",x,self.w1)+self.b1)
        h=torch.nn.functional.gelu(torch.einsum("bsh,shk->bsk",h,self.w2)+self.b2)
        return (h*self.w3).sum(-1)+self.b3


def factory(config, inputs, outputs):
    if config["arm"]=="shared":return head(config,inputs,outputs)
    if config["arm"]=="species":return SpeciesHeads(inputs,outputs,config["seed"]).to("cuda")
    raise ValueError("Unknown parameter-sharing arm")
