"""Fresh process, state-only input, complete live kinetic-feature inference."""
import argparse
import time
from pathlib import Path
import cantera as ct
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json, digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base, base_prediction
from ..adaptive.model import base_preprocessing
from ..resolved_fit.fit import head
from .model import predict, rate_features


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--seed",type=int,required=True); args=p.parse_args()
    env=configure(); rows=arrays("/input/states.npz"); plan=read_json("/input/config.json")
    if set(rows) != {"states","source_indices"}: raise ValueError("State-only boundary violated")
    base,prep=load_base(Path("/model"),args.seed); gas=ct.Solution("/model/mechanism.yaml")
    if gas.species_names != read_json("/model/species.json"): raise ValueError("Species mismatch")
    scaler=arrays("/model/scaler.npz"); saved=dict(source_indices=rows["source_indices"]); records=[]
    b,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed); saved["base"]=b
    for arm in plan["arms"]:
        config=read_json(Path("/model")/arm/"config.json")
        inputs=235 if arm=="kinetic" else 119
        model=head(config,inputs,int(prep["active"].sum()))
        model.load_state_dict(torch.load(Path("/model")/arm/"weights.pt",weights_only=True,map_location="cuda")); model.eval()
        pred,c=predict(base,model,prep,rows["states"],config,gas,scaler)
        saved[arm]=pred; saved[arm+"_coordinate"]=c
        # Fixed training-input prefix; warm end-to-end times, no cached rates.
        states=rows["states"][:1024]; timing=dict(anchor=[],candidate=[])
        for repeat in range(6):
            for name in ("anchor","candidate") if repeat%2==0 else ("candidate","anchor"):
                torch.cuda.synchronize(); start=time.monotonic()
                if name=="anchor":base_prediction(base,base_preprocessing(prep),states,args.seed)
                else:predict(base,model,prep,states,config,gas,scaler)
                torch.cuda.synchronize()
                if repeat:timing[name].append(time.monotonic()-start)
        records.append(dict(arm=arm,timings=timing,ratio=float(np.median(timing["candidate"])/np.median(timing["anchor"]))))
        del model; torch.cuda.empty_cache()
    rates,raw=rate_features(gas,rows["states"],prep["active"])
    saved["rate_coordinate"]=rates; saved["rate_raw"]=raw
    np.savez("/output/predictions.npz",**saved)
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,
        prediction_sha256=digest("/output/predictions.npz"),label_free_input=True))


if __name__=="__main__":main()
