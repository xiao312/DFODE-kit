"""Finite GPU calibration or isolated state-only replay."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,digest,read_json
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base,base_prediction
from ..adaptive.model import base_preprocessing
from .model import train_constants,calibrated_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","infer"]);p.add_argument("--arm",required=True);p.add_argument("--seed",type=int,required=True)
    args=p.parse_args();env=configure();destination=Path("/output");start=time.perf_counter()
    if args.mode=="train":
        config=read_json("/input/config.json")
        if config["seed"]!=args.seed or config["arm"]!=args.arm:raise ValueError("Configuration mismatch")
        rows=arrays("/input/training.npz")
        constants,pred,c,detail=train_constants(rows,args.arm)
        np.savez(destination/"constants.npz",constants=constants)
        reload=arrays(destination/"constants.npz")["constants"]
        np.testing.assert_array_equal(pred,calibrated_prediction(rows["base"],reload,rows["active"])[0])
        np.savez(destination/"training-replay.npz",prediction=pred,coordinate=c)
        save(destination/"result.json",dict(status="complete",environment=env,arm=args.arm,seed=args.seed,
             parameter_count=58,config=config,total_seconds=time.perf_counter()-start,detail=detail,constants_sha256=digest(destination/"constants.npz")))
        return
    rows=arrays("/input/states.npz")
    if set(rows)!={"states","source_indices"}:raise ValueError("State-only inference required")
    base,prep=load_base(Path("/model"),args.seed)
    constants=arrays("/model/constants.npz")["constants"]
    if constants.shape!=(58,) or not np.isfinite(constants).all():raise ValueError("Invalid artifact")
    def call(states):
        b,_=base_prediction(base,base_preprocessing(prep),states,args.seed)
        return calibrated_prediction(b,constants,prep["active"])
    pred,c=call(rows["states"]);again,ac=call(rows["states"])
    np.testing.assert_array_equal(pred,again);np.testing.assert_array_equal(c,ac)
    np.savez(destination/"predictions.npz",prediction=pred,coordinate=c,source_indices=rows["source_indices"])
    sample=rows["states"][:1023]
    calls=dict(candidate=lambda:call(sample),anchor=lambda:base_prediction(base,base_preprocessing(prep),sample,args.seed))
    times={name:[] for name in calls}
    for function in calls.values():function()
    for repeat in range(10):
        for name in (["candidate","anchor"] if repeat%2==0 else ["anchor","candidate"]):
            torch.cuda.synchronize();start=time.perf_counter();calls[name]();torch.cuda.synchronize()
            times[name].append(time.perf_counter()-start)
    save(destination/"result.json",dict(status="complete",environment=env,exact_repeat=True,
         states=len(pred),timing_states=len(sample),timings=times,prediction_sha256=digest(destination/"predictions.npz")))


if __name__=="__main__":main()
