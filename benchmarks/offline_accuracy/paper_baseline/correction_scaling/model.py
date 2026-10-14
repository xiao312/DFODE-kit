"""One label-free pipeline with fixed inference batch shapes across dataset sizes."""
import numpy as np
import torch
from benchmarks.flame_conditioning.train import network
from ..adaptive.model import features,base_preprocessing
from ..adaptive.plan import base_config
from ..matched_targets.model import prediction as original_prediction
from ..decoder_loss.coordinates import decode
from ..resolved_fit.fit import head


def fixed_batches(values,size=1024):
    if not len(values):
        raise ValueError("Cannot predict an empty population")
    for start in range(0,len(values),size):
        chunk = values[start:start+size]
        count = len(chunk)
        if count<size:
            chunk = np.concatenate([chunk,np.repeat(chunk[-1:],size-count,axis=0)])
        yield count,chunk


def base_prediction(base,bp,states,seed):
    values,masks = [],[]
    for count,chunk in fixed_batches(states):
        p,m = original_prediction(base,bp,chunk,base_config(seed))
        values.append(p[:count]); masks.append(m[:count])
    return np.concatenate(values),np.concatenate(masks)


def coordinates(model,x):
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(chunk,dtype=torch.float32,device="cuda")).double().cpu().numpy()[:n]
                               for n,chunk in fixed_batches(x)])


def cached_prediction(model,cache,active,config):
    c = coordinates(model,cache["features"])
    prediction = cache["base"].copy()
    prediction[:,active] = decode(prediction[:,active],c,config)
    return prediction,c


def prediction(base,model,prep,states,config):
    b,_ = base_prediction(base,base_preprocessing(prep),states,config["seed"])
    cache = dict(base=b,features=features(states,b,prep,config).astype(np.float32))
    return cached_prediction(model,cache,prep["active"],config)


def load_base(directory,seed):
    with np.load(directory/"preprocessing.npz") as arrays:
        prep = dict(arrays)
    base = network(len(prep["base_x_offset"]),int(prep["active"].sum()),[800]*4,seed,torch.float32,"gelu").to("cuda")
    base.load_state_dict(torch.load(directory/"base-weights.pt",weights_only=True,map_location="cuda"))
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base,prep


def load_head(directory,prep,config):
    model = head(config,len(prep["base_x_offset"])+int(prep["active"].sum()),int(prep["active"].sum()))
    model.load_state_dict(torch.load(directory/"weights.pt",weights_only=True,map_location="cuda"))
    return model.eval()
