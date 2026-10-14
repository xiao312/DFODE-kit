"""Fresh state-only replay and live end-to-end cost, including affinities."""
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
from .chemistry import metadata
from .model import ReactionHead, predict


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    env=configure();rows=arrays('/input/states.npz');plan=read_json('/input/config.json')
    if set(rows)!={'states','source_indices'}:raise ValueError('State-only boundary violated')
    base,prep=load_base(Path('/model'),args.seed);gas=ct.Solution('/model/mechanism.yaml')
    graph=metadata(gas,read_json('/model/species.json'));stored=arrays('/model/graph.npz')
    for key,value in graph.items():np.testing.assert_array_equal(value,stored[key])
    scaler=arrays('/model/scaler.npz');saved=dict(source_indices=rows['source_indices']);records=[]
    saved['base']=base_prediction(base,base_preprocessing(prep),rows['states'],args.seed)[0]
    for arm in plan['arms']:
        cfg=read_json(Path('/model')/arm/'config.json');model=ReactionHead(cfg,graph).to('cuda')
        model.load_state_dict(torch.load(Path('/model')/arm/'weights.pt',weights_only=True,map_location='cuda'));model.eval()
        pred,c=predict(base,model,prep,rows['states'],cfg,gas,scaler)
        saved[arm],saved[arm+'_coordinate']=pred,c
        states=rows['states'][:1023];timing=dict(anchor=[],candidate=[])
        for repeat in range(6):
            for name in ('anchor','candidate') if repeat%2==0 else ('candidate','anchor'):
                torch.cuda.synchronize();start=time.monotonic()
                if name=='anchor':base_prediction(base,base_preprocessing(prep),states,args.seed)
                else:predict(base,model,prep,states,cfg,gas,scaler)
                torch.cuda.synchronize()
                if repeat:timing[name].append(time.monotonic()-start)
        records.append(dict(arm=arm,timings=timing,ratio=float(np.median(timing['candidate'])/np.median(timing['anchor']))))
        del model;torch.cuda.empty_cache()
    np.savez('/output/predictions.npz',**saved)
    save(Path('/output/result.json'),dict(status='complete',records=records,environment=env,
        prediction_sha256=digest('/output/predictions.npz'),label_free_input=True))


if __name__=='__main__':main()
