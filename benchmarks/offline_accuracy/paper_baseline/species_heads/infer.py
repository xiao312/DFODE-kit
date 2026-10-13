"""State-only saved head replay and matched end-to-end timing."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base,base_prediction,prediction
from ..adaptive.model import base_preprocessing
from .model import factory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();rows=arrays("/input/states.npz");plan=read_json("/input/config.json")
    if set(rows)!={"states","source_indices"}:raise ValueError("Expected label-free states")
    base,prep=load_base(Path("/model"),args.seed);b,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed)
    values=dict(base=b,source_indices=rows["source_indices"]);records=[]
    for arm in plan["arms"]:
        config=read_json(Path("/model")/arm/"config.json");model=factory(config,119,58)
        model.load_state_dict(torch.load(Path("/model")/arm/"weights.pt",weights_only=True,map_location="cuda"));model.eval()
        values[arm],values[arm+"_coordinate"]=prediction(base,model,prep,rows["states"],config)
        states=rows["states"][:1024];timing=dict(anchor=[],candidate=[])
        for repeat in range(6):
            for name in ("anchor","candidate") if repeat%2==0 else ("candidate","anchor"):
                torch.cuda.synchronize();start=time.monotonic()
                if name=="anchor":base_prediction(base,base_preprocessing(prep),states,args.seed)
                else:prediction(base,model,prep,states,config)
                torch.cuda.synchronize()
                if repeat:timing[name].append(time.monotonic()-start)
        records.append(dict(arm=arm,timings=timing,ratio=float(np.median(timing["candidate"])/np.median(timing["anchor"]))))
        del model;torch.cuda.empty_cache()
    np.savez("/output/predictions.npz",**values)
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,prediction_sha256=digest("/output/predictions.npz")))


if __name__=="__main__":main()
