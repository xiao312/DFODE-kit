"""Freeze continued incumbents and emit coverage before any larger correction fit."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.evaluate import audit_subset
from ..matched_work.run import inputs,checked_result
from ..adaptive.model import reload_model,base_preprocessing,features
from ..adaptive.plan import configuration as continued_configuration
from ..adaptive.coordinates import preprocessing
from ..resolved_fit.data import selected_rows
from ..resolved_fit.run import require_idle
from ..decoder_loss.coordinates import encode
from .plan import SEEDS,BASE_HASHES,CONTINUED_HASHES,SIZES,configuration
from .model import base_prediction
from .coverage import profile,support,label_check


def checked_inputs(args,seed):
    path = args.continued_first if seed==SEEDS[0] else args.continued_second
    if sha256(path/"result.json") != CONTINUED_HASHES[seed]:
        raise ValueError("Continued incumbent changed")
    result = checked_result(path)
    if result["config"] != continued_configuration("continue",seed) or result["baseline_result_sha256"] != BASE_HASHES[seed]:
        raise ValueError("Continuation lineage differs")
    t,n,v,p,a,h = inputs(args.campaign,args.previous,args.original,args.base_root,"fuel-state",seed)
    if h!=result["hashes"]:
        raise ValueError("Data identity differs from continued base")
    np.testing.assert_array_equal(t["source_indices"],np.load(path/"training-indices.npy"))
    np.testing.assert_array_equal(n["source_indices"],np.load(path/"normalization-indices.npy"))
    return path,t,n,v,p,a,h


def cache(rows,b,prep,config):
    return dict(rows,base=b,features=features(rows["states"],b,prep,config).astype(np.float32),
                target=encode(b[:,prep["active"]],rows["delta"][:,prep["active"]],config))


def prepare_seed(args,seed):
    path,t,n,v,physics,audit,hashes = checked_inputs(args,seed)
    active = np.array([s!="AR" for s in physics["species_names"]])
    probe = selected_rows(t,audit,active)
    vi,vr,vu = audit_subset(v,audit)
    labels = dict(training=label_check(probe["ordinary_delta"],probe["reference"],probe["uncertainty"],active),
                  development=label_check(v["delta"][vi],vr,vu,active))
    if not all(q["passed"] for q in labels.values()):
        raise ValueError("Audited ordinary-label discrepancy exceeds 0.1 allowance")
    root = args.output/str(seed)
    root.mkdir()
    coverage = dict(development=profile(v,physics["species_names"]),training={})
    for size in SIZES:
        coverage["training"][str(size)] = profile({k:val[:size] for k,val in t.items()},physics["species_names"])
    coverage["support"] = {str(size):support(coverage["training"][str(size)],coverage["development"]) for size in SIZES}
    coverage["label_comparison"] = labels
    save(root/"coverage.json",coverage)
    models,old_prep = reload_model(path,continued_configuration("continue",seed))
    base = models[1].eval()
    for p in base.parameters():
        p.requires_grad_(False)
    bp = base_preprocessing(old_prep)
    config = configuration(seed,10000,"direct")
    btrain,_ = base_prediction(base,bp,t["states"],seed)
    bdev,_ = base_prediction(base,bp,v["states"],seed)
    prep = preprocessing(bp,btrain[:50000],n["delta"],config)
    np.savez(root/"preprocessing.npz",**prep)
    torch.save(base.state_dict(),root/"base-weights.pt")
    np.savez(root/"physics.npz",**physics)
    save(root/"audit.json",audit)
    replay = {}
    for split,rows,b in (("training",t,btrain),("development",v,bdev)):
        with np.load(path/f"{split}-predictions.npz") as arrays:
            old = arrays["prediction"]
            budget = 1e-15+.1*abs(rows["delta"])
            replay[split] = dict(max_change_in_allowances=float(np.max((abs(old-b)/budget)[:,active])),
                changed_pass_decisions=int((((abs(old-rows["delta"])<=budget)!=(abs(b-rows["delta"])<=budget))[:,active]).sum()))
        prepared = cache(rows,b,prep,config)
        if not np.isfinite(prepared["features"]).all() or not np.isfinite(prepared["target"]).all():
            raise ValueError("Nonfinite prepared learning arrays")
        np.savez(root/f"{split}.npz",**prepared)
    lookup = {int(s):i for i,s in enumerate(t["source_indices"])}
    indices = [lookup[int(s)] for s in probe["source_indices"]]
    bprobe,_ = base_prediction(base,bp,probe["states"],seed)
    np.testing.assert_array_equal(bprobe,btrain[indices])
    np.savez(root/"probe.npz",**probe,base=bprobe,features=features(probe["states"],bprobe,prep,config).astype(np.float32))
    metadata = dict(seed=seed,hashes=hashes,continued_result_sha256=CONTINUED_HASHES[seed],
        original_result_sha256=BASE_HASHES[seed],species=physics["species_names"],label_version="unchanged nominal endpoint labels",
        canonical_batch_comparison=replay,artifacts={p.name:sha256(p) for p in root.iterdir() if p.is_file()})
    save(root/"manifest.json",metadata)
    return dict(seed=seed,manifest_sha256=sha256(root/"manifest.json"),label_comparison=labels,
                canonical_batch_comparison=replay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","continued-first","continued-second","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new prepared directory")
    for seed in SEEDS:
        checked_inputs(args,seed)
    print(json.dumps(dict(status="planned",seeds=SEEDS,sizes=SIZES,base_hashes=CONTINUED_HASHES)),flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit scientific source first")
    require_idle()
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    status = dict(status="running",source=source,seeds=[])
    save(args.output/"status.json",status)
    try:
        for seed in SEEDS:
            record = prepare_seed(args,seed)
            status["seeds"].append(record)
            save(args.output/"status.json",status)
            print(json.dumps(record),flush=True)
        status["status"] = "complete"
        save(args.output/"status.json",status)
        save(args.output/"verification.json",dict(status="verified",status_sha256=sha256(args.output/"status.json")))
    except Exception as error:
        status.update(status="failed",error=str(error))
        save(args.output/"status.json",status)
        raise


if __name__=="__main__":
    main()
