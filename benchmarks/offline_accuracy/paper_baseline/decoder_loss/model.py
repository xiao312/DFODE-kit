"""Reuse identical correction heads/features; change only their output decoder."""
import numpy as np
import torch
from ..adaptive.model import create_models,reload_model,features,base_preprocessing
from ..matched_targets.model import prediction as base_prediction
from ..adaptive.plan import base_config
from .coordinates import decode


def prediction(models,prep,states,config):
    base,mask = base_prediction(models[0],base_preprocessing(prep),states,base_config(config["seed"]))
    x = features(states,base,prep,config)
    with torch.no_grad():
        parts = [models[1](torch.as_tensor(chunk,dtype=torch.float32,device="cuda")).double().cpu().numpy()
            for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))]
    active = prep["active"]
    output = base.copy()
    output[:,active] = decode(base[:,active],np.concatenate(parts),config)
    return output,mask
