"""Small bounded teacher head; one shared frozen forward at inference."""
import numpy as np
import torch
from benchmarks.flame_conditioning.coordinates import input_features
from ..adaptive.model import base_preprocessing
from ..adaptive.plan import base_config
from ..matched_targets.coordinates import decode as base_decode
from ..correction_scaling.model import fixed_batches, coordinates


def cap(base):
    return .1 * np.minimum(abs(base), 1e-15 + .1*abs(base))


def teacher_fraction(base, other):
    """Reference-free fraction of the permitted shrink movement."""
    movement=cap(base)
    toward_zero=np.sign(base)*(.5*base-.5*other)
    return np.clip(np.divide(toward_zero,movement,out=np.zeros_like(base),where=movement>0),0.,1.)


def decode(base, fraction):
    if not np.isfinite(fraction).all() or np.any(fraction<0) or np.any(fraction>1):
        raise ValueError("Fraction outside declared interval")
    proposed=base-np.sign(base)*cap(base)*fraction
    inner=.9*base
    return np.minimum(np.maximum(proposed,np.minimum(base,inner)),np.maximum(base,inner))


def objective(logits, base, truth, eligible, target, config):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.double()[eligible],target[eligible])


def cached_prediction(model,cache,active,config):
    logits=coordinates(model,cache["features"])
    # Use the same FP64 sigmoid in cached and fresh inference.
    fraction=torch.sigmoid(torch.from_numpy(logits)).numpy()
    result=cache["base"].copy();result[:,active]=decode(result[:,active],fraction)
    return result,fraction


def initial_check(prediction,cache,active,config):
    expected=cache["base"].copy();expected[:,active]=decode(expected[:,active],np.full_like(expected[:,active],.5))
    np.testing.assert_array_equal(prediction,expected)


def predict(base,head,prep,states,seed,constant=None):
    p=base_preprocessing(prep);active=prep["active"];outputs=[];bases=[];fractions=[]
    for count,chunk in fixed_batches(states):
        x=(input_features(chunk)-p["x_offset"])/p["x_scale"]
        with torch.no_grad():
            h=base[:-1](torch.as_tensor(x,dtype=torch.float32,device="cuda"))
            z=base[-1](h).double().cpu().numpy()*p["y_scale"]+p["y_offset"]
            if constant is None:
                logits=head(h).double().cpu()
                alpha=torch.sigmoid(logits).numpy()
            else:alpha=np.broadcast_to(constant,(len(chunk),len(constant)))
        d,_=base_decode(chunk[:,2:][:,active],z,base_config(seed))
        b=np.zeros_like(chunk[:,2:]);b[:,active]=d
        result=b.copy();result[:,active]=decode(d,alpha)
        outputs.append(result[:count]);bases.append(b[:count]);fractions.append(alpha[:count])
    return np.concatenate(outputs),np.concatenate(bases),np.concatenate(fractions)
