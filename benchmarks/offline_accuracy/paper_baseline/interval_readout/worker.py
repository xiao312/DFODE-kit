"""Reuse the paired optimizer loop; only target-set objective changes."""
import argparse
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json, digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..correction_scaling.plan import configuration
from ..correction_scaling.model import load_base, base_prediction
from ..correction_scaling.train import fit
from ..adaptive.model import base_preprocessing
from ..resolved_fit.fit import head
from ..hidden_features.prepare import save
from .model import collect, normalized_hidden, reconstruct, cached_prediction
from .targets import targets, objective


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--seed', type=int, required=True)
    seed = p.parse_args().seed; env = configure(); root = Path('/output')
    plan = read_json('/input/config.json'); rows = arrays('/input/training.npz')
    if len(rows['states']) != plan['fit_rows']: raise ValueError('Wrong training rows')
    base, prep = load_base(Path('/model'),seed); active = prep['active']; bp = base_preprocessing(prep)
    h, raw = collect(base,prep,rows['states']); b,_ = reconstruct(rows['states'],raw,prep,seed)
    np.testing.assert_array_equal(b,rows['base'])
    np.testing.assert_array_equal(b,base_prediction(base,bp,rows['states'],seed)[0])
    packed, alpha, details = targets(rows['states'][:,2:][:,active],rows['delta'][:,active],raw,bp)
    physical_pass = abs(b[:,active]-rows['delta'][:,active]) <= 1e-15+.1*abs(rows['delta'][:,active])
    coordinate_pass = (packed[...,0] <= 0) & (packed[...,1] >= 0)
    np.testing.assert_array_equal(physical_pass,coordinate_pass)
    scale = h.astype(float).std(0); constant = scale==0; scale[constant] = 1.
    scales = dict(hidden_offset=h.astype(float).mean(0),hidden_scale=scale,output_scale=alpha)
    np.savez(root/'scales.npz',**scales)
    x = normalized_hidden(torch.as_tensor(h,device='cuda'),scales).cpu().numpy()
    training = dict(states=rows['states'],features=x,base=b,delta=rows['delta'],target=packed,
                    diagnostic_target=packed[...,3],eligible=np.ones(physical_pass.shape,bool),
                    raw=raw,output_scale=alpha,preprocessing=prep)
    ix = np.linspace(0,len(h)-1,32,dtype=int)
    np.savez(root/'hidden-audit.npz',indices=ix,hidden=h[ix],features=x[ix],raw=raw[ix])
    np.savez(root/'targets.npz',target=packed,raw=raw,source_indices=rows['source_indices'])
    records = []
    for arm in plan['arms']:
        dest = root/arm; dest.mkdir()
        cfg = configuration(seed,10000,'coordinate-sl1')
        cfg.update(experiment=plan['experiment_id'],arm=arm,widths=[],training_count=len(x),updates=plan['batches'],
                   batch_size=plan['batch_size'],learning_rate=plan['learning_rate'],final_learning_rate=plan['final_learning_rate'],
                   wall_seconds=plan['resources']['maximum_fit_seconds'])
        save(dest/'config.json',cfg)
        model,record = fit(training,None,active,cfg,dest,objective_function=objective,prediction_function=cached_prediction)
        if record['parameter_count'] != plan['parameters']: raise ValueError('Parameter count')
        restored = head(cfg,800,58); restored.load_state_dict(torch.load(dest/'weights.pt',weights_only=True,map_location='cuda'))
        pred,c = cached_prediction(model,training,active,cfg)
        pred2,c2 = cached_prediction(restored,training,active,cfg)
        np.testing.assert_array_equal(pred,pred2); np.testing.assert_array_equal(c,c2)
        np.savez(dest/'training-replay.npz',prediction=pred,coordinate=c,source_indices=rows['source_indices'])
        record.update(arm=arm,exact_reload=True); records.append(record)
    if records[0]['initial_weights_sha256'] != records[1]['initial_weights_sha256']: raise ValueError('Initial heads differ')
    if any(p.requires_grad or p.grad is not None for p in base.parameters()): raise ValueError('Base not frozen')
    save(root/'result.json',dict(status='complete',environment=env,records=records,prechecks=details,
         hidden_constant_count=int(constant.sum()),initial_interval_pass_parity=True,scales_sha256=digest(root/'scales.npz')))


if __name__ == '__main__': main()
