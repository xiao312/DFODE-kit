"""Fit matched correction heads on the declared 20k training labels only."""
import argparse
import time
from pathlib import Path
import cantera as ct
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json, digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base, base_prediction, cached_prediction
from ..correction_scaling.plan import configuration
from ..correction_scaling.train import fit
from ..adaptive.model import features, base_preprocessing
from ..decoder_loss.coordinates import encode
from ..resolved_fit.fit import head
from .model import rate_features, fit_scaler, augment


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--seed",type=int,required=True); args=p.parse_args()
    env=configure(); started=time.monotonic(); plan=read_json("/input/config.json"); rows=arrays("/input/training.npz")
    if len(rows["states"]) != plan["fit_rows"]: raise ValueError("Unexpected training population")
    base,prep=load_base(Path("/model"),args.seed); active=prep["active"]
    gas=ct.Solution("/model/mechanism.yaml")
    if gas.species_names != read_json("/model/species.json"): raise ValueError("Mechanism species mismatch")
    cfg=configuration(args.seed,10000,"coordinate-sl1")
    b,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed)
    np.testing.assert_array_equal(b,rows["base"])
    ordinary=features(rows["states"],b,prep,cfg).astype(np.float32)
    np.testing.assert_array_equal(ordinary,rows["features"])
    rates,raw=rate_features(gas,rows["states"],active); scaler=fit_scaler(rates)
    np.savez("/output/scaler.npz",**scaler)
    np.savez("/output/rates.npz",coordinate=rates,raw=raw,source_indices=rows["source_indices"])
    records=[]
    for arm in plan["arms"]:
        destination=Path("/output")/arm; destination.mkdir()
        config=dict(cfg,experiment=plan["experiment_id"],arm=arm,widths=plan["widths"][arm],
            training_count=plan["fit_rows"],updates=plan["batches"],batch_size=plan["batch_size"],
            wall_seconds=plan["resources"]["maximum_fit_seconds"])
        save(destination/"config.json",config)
        x=augment(ordinary,rates,scaler) if arm=="kinetic" else ordinary
        training=dict(features=x,base=b,delta=rows["delta"],target=encode(b[:,active],rows["delta"][:,active],config))
        model,record=fit(training,None,active,config,destination)
        if record["parameter_count"] != plan["parameters"][arm]: raise ValueError("Parameter count mismatch")
        replay=head(config,x.shape[1],int(active.sum())); replay.load_state_dict(torch.load(destination/"weights.pt",weights_only=True,map_location="cuda"))
        predicted,coordinate=cached_prediction(model,training,active,config)
        again,q=cached_prediction(replay,training,active,config)
        np.testing.assert_array_equal(predicted,again); np.testing.assert_array_equal(coordinate,q)
        np.savez(destination/"training-replay.npz",prediction=predicted,coordinate=coordinate,source_indices=rows["source_indices"])
        record.update(arm=arm,weights_sha256=digest(destination/"weights.pt")); records.append(record)
        del model,replay; torch.cuda.empty_cache()
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,
         seconds=time.monotonic()-started,exact_replay=True,candidate=False))


if __name__=="__main__":main()
