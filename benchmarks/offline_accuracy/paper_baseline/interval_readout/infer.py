"""Label-free saved-head replay and end-to-end paired timing."""
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);seed=p.parse_args().seed
    env=configure();started=time.monotonic();rows=arrays('/input/states.npz');plan=read_json('/input/config.json')
    if set(rows)!={'states','source_indices'}:raise ValueError('State-only inference required')
    base,prep=load_base(Path('/model'),seed);scales=arrays('/model/scales.npz');saved=dict(source_indices=rows['source_indices']);records=[]
    for arm in plan['arms']:
        cfg=read_json(Path('/model')/arm/'config.json');model=head(cfg,800,58)
        model.load_state_dict(torch.load(Path('/model')/arm/'weights.pt',weights_only=True,map_location='cuda'));model.eval()
        prediction,b,c,mask,base_mask=predict(base,model,prep,scales,rows['states'],seed)
        restored=head(cfg,800,58);restored.load_state_dict(torch.load(Path('/model')/arm/'weights.pt',weights_only=True,map_location='cuda'));restored.eval()
        replay=predict(base,restored,prep,scales,rows['states'],seed)
        for first,second in zip((prediction,b,c,mask,base_mask),replay):np.testing.assert_array_equal(first,second)
        historical,_=base_prediction(base,base_preprocessing(prep),rows['states'],seed);np.testing.assert_array_equal(b,historical)
        saved.update({arm:prediction,arm+'_coordinate':c,arm+'_domain_correction':mask,'base':b,'base_domain_correction':base_mask})
        timings=dict(anchor=[],candidate=[]);sample=rows['states'][:1024]
        for repeat in range(11):
            for name in ('anchor','candidate') if repeat%2==0 else ('candidate','anchor'):
                torch.cuda.synchronize();t=time.monotonic()
                if name=='anchor':base_prediction(base,base_preprocessing(prep),sample,seed)
                else:predict(base,model,prep,scales,sample,seed)
                torch.cuda.synchronize()
                if repeat:timings[name].append(time.monotonic()-t)
        records.append(dict(arm=arm,timings=timings,ratio=float(np.median(timings['candidate'])/np.median(timings['anchor']))))
    if time.monotonic()-started>plan['resources']['maximum_worker_seconds']:raise TimeoutError('Inference budget')
    np.savez('/output/predictions.npz',**saved)
    save(Path('/output/result.json'),dict(status='complete',environment=env,records=records,exact_reload_replay=True,
         prediction_sha256=digest('/output/predictions.npz')))


if __name__=='__main__':main()
