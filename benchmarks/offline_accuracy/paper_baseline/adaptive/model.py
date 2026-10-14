"""Self-contained frozen-base plus correction checkpoints and label-free inference."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from benchmarks.flame_conditioning.train import network
from ..matched_targets.model import prediction as base_prediction
from .plan import base_config
from .coordinates import scales


def base_preprocessing(prep):
    return {key[5:]:value for key,value in prep.items() if key.startswith("base_")}


def features(states, base, prep, config):
    p = base_preprocessing(prep)
    original = (input_features(states)-p["x_offset"])/p["x_scale"]
    extra = (np.arcsinh(base[:,prep["active"]]/config["scale_floor"])-prep["extra_offset"])/prep["extra_scale"]
    return np.column_stack([original,extra])


def prediction(models, prep, states, config):
    frozen, fitted = models
    p = base_preprocessing(prep)
    if config["arm"] == "continue":
        return base_prediction(fitted,p,states,base_config(config["seed"]))
    base, correction = base_prediction(frozen,p,states,base_config(config["seed"]))
    x = features(states,base,prep,config)
    with torch.no_grad():
        parts = [fitted(torch.as_tensor(chunk,dtype=torch.float32,device="cuda")).double().cpu().numpy()
                 for chunk in np.array_split(x,max(1,int(np.ceil(len(x)/1024))))]
    output = base.copy()
    active = prep["active"]
    output[:,active] += scales(base[:,active],prep,config)*np.concatenate(parts)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite correction; no clipping fallback")
    return output, correction


def create_models(base, prep, config):
    import copy
    frozen = copy.deepcopy(base).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    if config["arm"] == "continue":
        fitted = copy.deepcopy(base)
        for parameter in fitted.parameters():
            parameter.requires_grad_(True)
    else:
        fitted = network(len(prep["base_x_offset"])+int(prep["active"].sum()),int(prep["active"].sum()),
                         config["widths"],config["seed"],torch.float32,"gelu").to("cuda")
        with torch.no_grad():
            fitted[-1].weight.zero_()
            fitted[-1].bias.zero_()
    return frozen, fitted


def reload_model(directory,config):
    with np.load(directory/"preprocessing.npz",allow_pickle=False) as saved:
        prep = dict(saved)
    p = base_preprocessing(prep)
    base = network(len(p["x_offset"]),int(p["active"].sum()),[800]*4,config["seed"],torch.float32,"gelu").to("cuda")
    base.load_state_dict(torch.load(directory/"base-weights.pt",weights_only=True,map_location="cuda"))
    models = create_models(base,prep,config)
    models[1].load_state_dict(torch.load(directory/"weights.pt",weights_only=True,map_location="cuda"))
    models[1].eval()
    return models,prep
