"""State-only projection and exact replay, with complete inference timing."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base,base_prediction
from ..adaptive.model import base_preprocessing
from .model import project


def predict(base,prep,states,seed,matrix,arm):
    b,_=base_prediction(base,base_preprocessing(prep),states,seed)
    active=prep["active"];pred=b.copy();flags=[]
    for start in range(0,len(b),1024):
        out,fallback=project(torch.as_tensor(b[start:start+1024,active],device="cuda"),matrix,arm)
        pred[start:start+1024,active]=out.cpu().numpy();flags.append(fallback.cpu().numpy())
    return pred,np.concatenate(flags),b


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();started=time.monotonic();plan=read_json("/input/config.json");rows=arrays("/input/states.npz")
    if set(rows)!={"states","source_indices"}:raise ValueError("Inference inputs contain unexpected arrays")
    base,prep=load_base(Path("/model"),args.seed);matrix=torch.as_tensor(np.load("/model/element-matrix.npy"),device="cuda")
    saved=dict(source_indices=rows["source_indices"]);records=[]
    for arm in plan["arms"]:
        pred,flags,b=predict(base,prep,rows["states"],args.seed,matrix,arm)
        again,f,bb=predict(base,prep,rows["states"],args.seed,matrix,arm)
        np.testing.assert_array_equal(pred,again);np.testing.assert_array_equal(flags,f);np.testing.assert_array_equal(b,bb)
        saved[arm]=pred;saved[arm+"_fallback"]=flags;saved["base"]=b
        timings=dict(anchor=[],candidate=[]);states=rows["states"][:1024]
        for repeat in range(6):
            for name in ("anchor","candidate") if repeat%2==0 else ("candidate","anchor"):
                torch.cuda.synchronize();start=time.monotonic()
                if name=="anchor":base_prediction(base,base_preprocessing(prep),states,args.seed)
                else:predict(base,prep,states,args.seed,matrix,arm)
                torch.cuda.synchronize()
                if repeat:timings[name].append(time.monotonic()-start)
        if time.monotonic()-started>plan["resources"]["maximum_worker_seconds"]:raise TimeoutError("Finite projection work exceeded budget")
        records.append(dict(arm=arm,fallback_states=int(flags.sum()),timings=timings,ratio=float(np.median(timings["candidate"])/np.median(timings["anchor"]))))
    np.savez("/output/predictions.npz",**saved)
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,seconds=time.monotonic()-started,exact_replay=True,prediction_sha256=digest("/output/predictions.npz"),candidate=False))


if __name__=="__main__":main()
