"""Paired GPU fits. Targets enter only this training worker, not inference."""
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
from ..correction_scaling.plan import configuration
from ..correction_scaling.train import fit
from ..adaptive.model import features, base_preprocessing
from ..decoder_loss.coordinates import encode
from .chemistry import metadata, affinity, fit_scaler, pack
from .model import ReactionHead, cached_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    env=configure();start=time.monotonic();plan=read_json('/input/config.json');rows=arrays('/input/training.npz')
    if len(rows['states'])!=plan['fit_rows']:raise ValueError('Training population differs')
    base,prep=load_base(Path('/model'),args.seed);active=prep['active']
    gas=ct.Solution('/model/mechanism.yaml');graph=metadata(gas,read_json('/model/species.json'))
    np.testing.assert_array_equal(active,graph['active']);np.savez('/output/graph.npz',**graph)
    cfg=configuration(args.seed,10000,'coordinate-sl1')
    b,_=base_prediction(base,base_preprocessing(prep),rows['states'],args.seed)
    np.testing.assert_array_equal(b,rows['base'])
    ordinary=features(rows['states'],b,prep,cfg).astype(np.float32)
    started=time.monotonic();raw=affinity(gas,rows['states']);scaler=fit_scaler(raw)
    x=pack(ordinary,raw,scaler);feature_seconds=time.monotonic()-started
    np.savez('/output/scaler.npz',**scaler)
    np.savez('/output/feature-audit.npz',raw_affinity=raw,ordinary=ordinary,source_indices=rows['source_indices'])
    training=dict(features=x,base=b,delta=rows['delta'],target=encode(b[:,active],rows['delta'][:,active],cfg))
    records=[]
    for arm in plan['arms']:
        dest=Path('/output')/arm;dest.mkdir()
        config=dict(cfg,experiment=plan['experiment_id'],arm=arm,training_count=plan['fit_rows'],
            updates=plan['batches'],batch_size=plan['batch_size'],width=plan['width'],heads=plan['heads'],
            diagnostics_every_updates=plan['diagnostics_every_updates'],learning_rate=plan['learning_rate'],
            final_learning_rate=plan['final_learning_rate'],wall_seconds=plan['resources']['maximum_fit_seconds'])
        save(dest/'config.json',config)
        factory=lambda c,_inputs,_outputs:ReactionHead(c,graph).to('cuda')
        model,record=fit(training,None,active,config,dest,model_factory=factory,prediction_function=cached_prediction)
        replay=factory(config,None,None);replay.load_state_dict(torch.load(dest/'weights.pt',weights_only=True,map_location='cuda'))
        pred,c=cached_prediction(model,training,active,config);again,q=cached_prediction(replay,training,active,config)
        np.testing.assert_array_equal(pred,again);np.testing.assert_array_equal(c,q)
        np.savez(dest/'training-replay.npz',prediction=pred,coordinate=c,source_indices=rows['source_indices'])
        record.update(arm=arm,exact_reload=True,weights_sha256=digest(dest/'weights.pt'));records.append(record)
        del model,replay;torch.cuda.empty_cache()
    save(Path('/output/result.json'),dict(status='complete',records=records,environment=env,
        feature_seconds=feature_seconds,seconds=time.monotonic()-start,confirmation_opened=False))


if __name__=='__main__':main()
