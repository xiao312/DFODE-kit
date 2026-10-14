"""Finite GPU scalar fits; only permitted training data is mounted."""
import argparse
import time
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..gradient_probe.metrics import physical_counts,groups
from .model import fit_map,predict,fold_ids


def summary(pred,d,b,initial):
    masks=groups(d,b);nt=abs(d)>1e-15/.9
    return dict(groups=physical_counts(pred,d,b,masks),
                sign_errors=int(((np.sign(pred)!=np.sign(d))&nt).sum()),
                negative=int(((initial+pred)<0).sum()),
                p99=float(np.quantile(abs(pred-d)/(1e-15+.1*abs(d)),.99)),
                species_passes=(abs(pred-d)<=1e-15+.1*abs(d)).sum(0).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();started=time.monotonic();plan=read_json("/input/config.json")
    if plan["folds"]!=2 or plan["fold_seed_offset"]!=411:raise ValueError("Unimplemented fold plan")
    data=arrays("/input/training.npz");prep=arrays("/model/preprocessing.npz");active=prep["active"]
    d=data["delta"][:,active];b=data["base"][:,active];initial=data["states"][:,2:][:,active]
    fold=fold_ids(len(d),args.seed);saved=dict(source_indices=data["source_indices"],fold=fold,truth=d,baseline=b,initial=initial)
    records=[];artifacts={};deadline=started+plan["resources"]["maximum_worker_seconds"]
    for arm in plan["arms"]:
        allpred=np.empty_like(b);models=[];fits=[]
        for held in range(2):
            fitting=fold!=held;checking=~fitting
            model=fit_map(b[fitting],d[fitting],plan["bins"] if arm=="conditional" else 1,
                          plan["minimum_bin_count"],deadline=deadline)
            fitted=predict(b[fitting],model);allpred[checking]=predict(b[checking],model)
            models.append(model);fits.append(summary(fitted,d[fitting],b[fitting],initial[fitting]))
        save(Path("/output")/(arm+"-maps.json"),models)
        replay=read_json(Path("/output")/(arm+"-maps.json"))
        for held in range(2):np.testing.assert_array_equal(predict(b[fold==held],replay[held]),allpred[fold==held])
        saved[arm]=allpred
        records.append(dict(arm=arm,fitting= fits,crossfit=summary(allpred,d,b,initial)))
    np.savez(Path("/output/predictions.npz"),**saved)
    save(Path("/output/result.json"),dict(status="complete",records=records,base=summary(b,d,b,initial),
         environment=env,seconds=time.monotonic()-started,prediction_sha256=digest("/output/predictions.npz"),
         active_species_indices=np.flatnonzero(active).tolist(),replay="exact",candidate=False))


if __name__=="__main__":main()
