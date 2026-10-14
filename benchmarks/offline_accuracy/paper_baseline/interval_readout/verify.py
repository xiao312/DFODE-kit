"""Additional denominator/raw-logit checks on immutable finished fits."""
import argparse
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,required=True)
    project=p.parse_args().project;root=project/'runs/offline-accuracy/interval-readout-20261013'
    out=root/'independent-denominators.json'
    if out.exists():raise ValueError('Check exists')
    review=read_json(root/'review.json');report=read_json(checked_file(root/'report.json',review['report_sha256']))
    for name,sha in report['artifacts'].items():checked_file(root/name,sha)
    checks=[]
    for seed in (20261011,20261012):
        d=root/str(seed);cache=arrays(d/'targets.npz');audit=arrays(d/'hidden-audit.npz');scale=arrays(d/'scales.npz')
        prep=arrays(root/'prepared'/str(seed)/'model/preprocessing.npz');rows=arrays(root/'prepared'/str(seed)/'training-reference.npz')
        active=prep['active'];y=rows['states'][:,2:][:,active];raw=cache['raw'];t=cache['target'];alpha=scale['output_scale']
        np.testing.assert_array_equal(audit['raw'],raw[audit['indices']])
        lo=raw+alpha*t[...,0];hi=raw+alpha*t[...,1]
        zero_endpoint=-2*np.sqrt(y**.1/.1/1e-6)
        zero_normalized=(zero_endpoint-prep['base_y_offset'])/prep['base_y_scale']
        lo=np.where(np.isneginf(lo),zero_normalized,lo)
        width=np.maximum((hi-lo)/2,128*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(lo),abs(hi))))
        error=float(np.max(abs(t[...,2]*alpha-width)/width))
        assert error<1e-8
        weights={k:v.double().numpy() for k,v in torch.load(root/'prepared'/str(seed)/'model/base-weights.pt',weights_only=True,map_location='cpu').items()}
        z=audit['hidden'].astype(float)@weights['8.weight'].T+weights['8.bias']
        np.testing.assert_allclose(z,audit['raw'],rtol=3e-5,atol=3e-6)
        checks.append(dict(seed=seed,width_relative_difference=error,raw_logit_max_difference=float(abs(z-audit['raw']).max())))
    save(out,dict(status='verified',review_sha256=digest(root/'review.json'),checks=checks,
                  limit='Hidden normalization application checked on32samples; full50k fitted hidden statistics source-enforced, not independently recomputed.'))
    print(digest(root/'review.json'));print(digest(out));print(checks)


if __name__=='__main__':main()
