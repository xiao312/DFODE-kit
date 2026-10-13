"""Existing base trainer with isolated training or state-only inference."""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..adaptive.fit import fit
from ..adaptive.model import base_preprocessing
from ..correction_scaling.model import load_base,base_prediction
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["train","infer"])
    p.add_argument("--arm",required=True);p.add_argument("--seed",type=int,required=True)
    args=p.parse_args();env=configure();dest=Path("/output");started=time.perf_counter()
    if args.mode=="train":
        cfg=read_json("/input/config.json")
        if (cfg["seed"],cfg["comparison_arm"])!=(args.seed,args.arm):raise ValueError("Configuration mismatch")
        rows=arrays("/input/training.npz");base,prep=load_base(Path("/input/model"),args.seed);bp=base_preprocessing(prep)
        np.testing.assert_array_equal(base_prediction(base,bp,rows["states"],args.seed)[0],rows["base"])
        models,frozen_prep,detail=fit(base,bp,rows,None,rows,cfg,dest,fixed_preprocessing=prep)
        for key,value in prep.items():np.testing.assert_array_equal(frozen_prep[key],value)
        prediction,_=base_prediction(models[1],bp,rows["states"],args.seed)
        replay=copy.deepcopy(base);replay.load_state_dict(torch.load(dest/"weights.pt",weights_only=True,map_location="cuda"))
        np.testing.assert_array_equal(prediction,base_prediction(replay,bp,rows["states"],args.seed)[0])
        np.savez(dest/"training-replay.npz",prediction=prediction,coordinate=np.empty((len(prediction),0)))
        detail["history"]=[dict(row,update=row["updates"]) for row in detail["history"]]
        save(dest/"result.json",dict(status="complete",config=cfg,environment=env,**detail,total_seconds=time.perf_counter()-started,
             preprocessing_sha256=digest("/input/model/preprocessing.npz"),exact_reload=True,training_only=True))
        return
    rows=arrays("/input/states.npz")
    if set(rows)!={"states","source_indices"}:raise ValueError("State-only inputs required")
    anchor,prep=load_base(Path("/model"),args.seed);bp=base_preprocessing(prep)
    candidate=copy.deepcopy(anchor);candidate.load_state_dict(torch.load("/model/weights.pt",weights_only=True,map_location="cuda"))
    def call(states):return base_prediction(candidate,bp,states,args.seed)[0]
    prediction=call(rows["states"]);np.testing.assert_array_equal(prediction,call(rows["states"]))
    np.savez(dest/"predictions.npz",prediction=prediction,coordinate=np.empty((len(prediction),0)),source_indices=rows["source_indices"])
    sample=rows["states"][:1023];calls=dict(candidate=lambda:call(sample),anchor=lambda:base_prediction(anchor,bp,sample,args.seed))
    times={name:[] for name in calls}
    for function in calls.values():function()
    for repeat in range(10):
        for name in (["candidate","anchor"] if repeat%2==0 else ["anchor","candidate"]):
            torch.cuda.synchronize();start=time.perf_counter();calls[name]();torch.cuda.synchronize()
            times[name].append(time.perf_counter()-start)
    save(dest/"result.json",dict(status="complete",environment=env,exact_repeat=True,states=len(prediction),timing_states=len(sample),
         timings=times,prediction_sha256=digest(dest/"predictions.npz")))


if __name__=="__main__":main()
