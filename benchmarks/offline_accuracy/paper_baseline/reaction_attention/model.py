"""Small species-query/reaction-key model with a matched fixed-pooling control."""
import numpy as np
import torch
from torch import nn
from .chemistry import normalize_columns, affinity, pack
from ..adaptive.model import features, base_preprocessing
from ..correction_scaling.model import base_prediction, fixed_batches
from ..decoder_loss.coordinates import decode


class ReactionHead(nn.Module):
    def __init__(self, config, graph):
        super().__init__()
        torch.manual_seed(config['seed'])
        width = config['width']; self.heads = config['heads']
        if width % self.heads:
            raise ValueError('Width must divide into attention heads')
        self.width = width; self.mode, extra = config['arm'].split('-')
        if self.mode not in ('pool', 'attention') or extra not in ('state', 'affinity'):
            raise ValueError('Unknown reaction model arm')
        self.use_affinity = extra == 'affinity'
        n, reactions = graph['reactant'].shape
        self.n_species = n; self.n_reactions = reactions
        active = np.flatnonzero(graph['active']); self.n_active = len(active)
        self.register_buffer('active', torch.as_tensor(active, dtype=torch.long))
        for name in ('reactant', 'product', 'collider'):
            self.register_buffer(name, torch.as_tensor(normalize_columns(graph[name]), dtype=torch.float32))
        self.register_buffer('constants', torch.as_tensor(graph['constants'], dtype=torch.float32))
        pooling = torch.as_tensor(graph['pooling'][active], dtype=torch.float32)
        self.register_buffer('pooling', pooling)
        self.register_buffer('log_prior', torch.where(pooling>0, pooling.log(), -torch.inf))
        self.species_embedding = nn.Parameter(torch.randn(self.n_active, width)*.02)
        self.reaction_embedding = nn.Parameter(torch.randn(reactions, width)*.02)
        self.species_input = nn.Linear(4, width)
        self.reaction_input = nn.Linear(9+graph['constants'].shape[1], width)
        self.value = nn.Linear(width, width)
        self.query_residual = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.norm2 = nn.LayerNorm(width)
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        # Initialize common modules first, preserving identical common weights
        # across both architecture arms and exact identities across affinity arms.
        if self.mode == 'attention':
            self.query = nn.Linear(width, width)
            self.key = nn.Linear(width, width)

    def tokens(self, x):
        expected = 2+self.n_species+self.n_active+self.n_reactions
        if x.ndim != 2 or x.shape[1] != expected:
            raise ValueError('Prepared feature axes differ')
        y = x[:, 2:2+self.n_species]
        b = torch.zeros_like(y)
        b[:, self.active] = x[:, 2+self.n_species:2+self.n_species+self.n_active]
        tp = x[:, :2]
        q = torch.stack([y[:, self.active], b[:, self.active],
                         tp[:, 0, None].expand(-1,self.n_active),
                         tp[:, 1, None].expand(-1,self.n_active)], dim=-1)
        pieces = [y@self.reactant, y@self.product, y@self.collider,
                  b@self.reactant, b@self.product, b@self.collider,
                  tp[:,0,None].expand(-1,self.n_reactions), tp[:,1,None].expand(-1,self.n_reactions)]
        a = x[:, -self.n_reactions:]
        pieces.append(a if self.use_affinity else torch.zeros_like(a))
        r = torch.cat([torch.stack(pieces, dim=-1), self.constants[None].expand(len(x),-1,-1)], dim=-1)
        return (torch.nn.functional.gelu(self.species_input(q)+self.species_embedding),
                torch.nn.functional.gelu(self.reaction_input(r)+self.reaction_embedding))

    def attention_weights(self, q, r):
        batch = len(q); part = self.width//self.heads
        query = self.query(q).reshape(batch,self.n_active,self.heads,part).transpose(1,2)
        key = self.key(r).reshape(batch,self.n_reactions,self.heads,part).transpose(1,2)
        return (query@key.transpose(-1,-2)/np.sqrt(part)+self.log_prior).softmax(-1)

    def forward(self, x):
        q,r = self.tokens(x); value = self.value(r)
        if self.mode == 'pool':
            message = torch.einsum('sr,brd->bsd', self.pooling, value)
        else:
            value = value.reshape(len(x),self.n_reactions,self.heads,self.width//self.heads).transpose(1,2)
            message = (self.attention_weights(q,r)@value).transpose(1,2).reshape(len(x),self.n_active,self.width)
        h = self.norm1(self.query_residual(q)+message)
        return self.output(self.norm2(h+self.ff(h))).squeeze(-1)


def coordinates(model, x):
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(chunk,device='cuda',dtype=torch.float32)).double().cpu().numpy()[:n]
                               for n,chunk in fixed_batches(x,128)])


def cached_prediction(model, cache, active, config):
    c = coordinates(model, cache['features']); out = cache['base'].copy()
    out[:,active] = decode(out[:,active],c,config)
    return out,c


def predict(base, model, prep, states, config, gas, scaler):
    b,_ = base_prediction(base,base_preprocessing(prep),states,config['seed'])
    ordinary = features(states,b,prep,config).astype(np.float32)
    # State-only arms never compute chemical potentials at inference.
    raw = affinity(gas,states) if model.use_affinity else np.zeros((len(states),gas.n_reactions))
    return cached_prediction(model,dict(base=b,features=pack(ordinary,raw,scaler)),prep['active'],config)
