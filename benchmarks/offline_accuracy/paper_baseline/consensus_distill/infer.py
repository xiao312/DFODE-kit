"""Head or constant inference with one base only and complete matched timing."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..correction_scaling.model import load_base,base_prediction
from ..adaptive.model import base_preprocessing
from ..resolved_fit.fit import head
from ..hidden_features.prepare import save
from .model import predict


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--seed",type=int,required=True);args=parser.parse_args()
    env=configure();started=time.monotonic();rows=arrays("/input/states.npz");plan=read_json("/input/config.json");cfg=read_json("/model/config.json")
    if set(rows)!={"states","source_indices"}:raise ValueError("Unexpected inference inputs")
    base,prep=load_base(Path("/model"),args.seed);model=head(cfg,800,58)
    model.load_state_dict(torch.load("/model/weights.pt",weights_only=True,map_location="cuda"));model.eval()
    constant=np.load("/model/constant.npy");saved=dict(source_indices=rows["source_indices"]);records=[]
    for arm in plan["arms"]:
        fraction=constant if arm=="constant" else None
        value,b,alpha=predict(base,model,prep,rows["states"],args.seed,fraction)
        restored=head(cfg,800,58);restored.load_state_dict(torch.load("/model/weights.pt",weights_only=True,map_location="cuda"));restored.eval()
        again,bb,aa=predict(base,restored,prep,rows["states"],args.seed,fraction)
        for left,right in ((value,again),(b,bb),(alpha,aa)):np.testing.assert_array_equal(left,right)
        fresh,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed);np.testing.assert_array_equal(fresh,b)
        saved[arm]=value;saved[arm+"_fraction"]=alpha;saved["base"]=b
        timings=dict(anchor=[],candidate=[]);states=rows["states"][:1024]
        for repeat in range(6):
            for name in ("anchor","candidate") if repeat%2==0 else ("candidate","anchor"):
                torch.cuda.synchronize();start=time.monotonic()
                if name=="anchor":base_prediction(base,base_preprocessing(prep),states,args.seed)
                else:predict(base,model,prep,states,args.seed,fraction)
                torch.cuda.synchronize()
                if repeat:timings[name].append(time.monotonic()-start)
        records.append(dict(arm=arm,timings=timings,ratio=float(np.median(timings["candidate"])/np.median(timings["anchor"]))))
    if time.monotonic()-started>plan["resources"]["maximum_worker_seconds"]:raise TimeoutError("Inference work exceeded budget")
    np.savez("/output/predictions.npz",**saved)
    save("/output/result.json",dict(status="complete",records=records,exact_reload_replay=True,environment=env,prediction_sha256=digest("/output/predictions.npz")))


if __name__=="__main__":main()
