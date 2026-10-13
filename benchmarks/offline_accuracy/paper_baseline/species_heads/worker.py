"""Existing fixed-work trainer, changed parameter sharing only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.plan import configuration
from ..correction_scaling.model import load_base,base_prediction,cached_prediction
from ..correction_scaling.train import fit
from ..adaptive.model import features,base_preprocessing
from ..decoder_loss.coordinates import encode
from .model import factory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();plan=read_json("/input/config.json");rows=arrays("/input/training.npz")
    if len(rows["states"])!=plan["fit_rows"]:raise ValueError("Unexpected fitting rows")
    base,prep=load_base(Path("/model"),args.seed);active=prep["active"];cfg=configuration(args.seed,10000,"coordinate-sl1")
    b,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed);np.testing.assert_array_equal(b,rows["base"])
    x=features(rows["states"],b,prep,cfg).astype(np.float32);np.testing.assert_array_equal(x,rows["features"])
    training=dict(base=b,delta=rows["delta"],features=x,target=encode(b[:,active],rows["delta"][:,active],cfg));records=[]
    for arm in plan["arms"]:
        config=dict(cfg,experiment=plan["experiment_id"],arm=arm,widths=[452,452],training_count=20000,
            updates=plan["batches"],batch_size=plan["batch_size"],wall_seconds=plan["resources"]["maximum_fit_seconds"])
        dest=Path("/output")/arm;dest.mkdir();save(dest/"config.json",config)
        model,record=fit(training,None,active,config,dest,model_factory=factory)
        if record["parameter_count"]!=plan["parameters"][arm]:raise ValueError("Wrong parameter count")
        restored=factory(config,119,58);restored.load_state_dict(torch.load(dest/"weights.pt",weights_only=True,map_location="cuda"))
        pred,c=cached_prediction(model,training,active,config);again,q=cached_prediction(restored,training,active,config)
        np.testing.assert_array_equal(pred,again);np.testing.assert_array_equal(c,q)
        np.savez(dest/"training-replay.npz",prediction=pred,coordinate=c,source_indices=rows["source_indices"])
        records.append(dict(record,arm=arm));del model,restored;torch.cuda.empty_cache()
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,exact_replay=True,candidate=False))


if __name__=="__main__":main()
