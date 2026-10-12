"""Validate pinned inputs, then fit only existing independently audited training rows."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.paired.fit import weight_hash
from ..matched_work.run import inputs, checked_result
from ..adaptive.plan import SEEDS, BASE_HASHES, CAMPAIGN_HASH, base_config
from ..adaptive.gpu import inventory, idle
from ..adaptive.model import features
from ..adaptive.coordinates import preprocessing
from ..matched_targets.model import reload_model as load_base, prediction as base_prediction
from ..decoder_loss.diagnostics import oracle
from .plan import configuration, WIDTHS, OBJECTIVES
from .data import selected_rows, scores
from .fit import fit


def load_inputs(args,seed):
    path = args.baseline/f"{seed}-gbct-increment"
    if sha256(args.baseline/"verification.json") != CAMPAIGN_HASH or sha256(path/"result.json") != BASE_HASHES[seed]:
        raise ValueError("Pinned base identity changed")
    baseline = checked_result(path)
    training,normalization,development,physics,audit,hashes = inputs(
        args.campaign,args.previous,args.original,args.base_root,"fuel-state",seed)
    # The shared gate reads development to verify dataset identity, not to fit or score.
    del development
    if baseline["config"] != base_config(seed) or baseline["hashes"] != hashes:
        raise ValueError("Frozen baseline configuration or data differs")
    active = np.asarray([name != "AR" for name in physics["species_names"]])
    sample = selected_rows(training,audit,active)
    return normalization,physics,hashes,sample,active


def require_idle():
    import os
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = next((g for g in inventory() if g["index"] == selected),None)
    if gpu is None or not idle(gpu):
        raise ValueError("Select a single idle GPU; do not interrupt other processes")
    return gpu


def execute(args,plans):
    import torch
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit scientific source before execution")
    gpu = require_idle()
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    status = dict(status="running",source=source,gpu=gpu,plan=plans,
                  development_count=0,independent_test_count=0,completed=[])
    save(args.output/"status.json",status)
    try:
        common_sample = None
        for seed in SEEDS:
            normalization,physics,hashes,sample,active = load_inputs(args,seed)
            if common_sample is not None:
                for key in sample:
                    np.testing.assert_array_equal(sample[key],common_sample[key])
            common_sample = sample
            root = args.output/str(seed)
            root.mkdir()
            config = configuration(seed,WIDTHS[0],OBJECTIVES[0])
            base,bp = load_base(args.baseline/f"{seed}-gbct-increment",base_config(seed))
            digest = weight_hash(base)
            nv,_ = base_prediction(base,bp,normalization["states"],base_config(seed))
            prep = preprocessing(bp,nv,normalization["delta"],config)
            bv,_ = base_prediction(base,bp,sample["states"],base_config(seed))
            x = features(sample["states"],bv,prep,config).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite network features")
            _,counts = np.unique(x,axis=0,return_counts=True)
            np.savez(root/"sample.npz",**sample,features=x,base=bv,active=active)
            np.savez(root/"preprocessing.npz",**prep)
            torch.save(base.state_dict(),root/"base-weights.pt")
            audit = dict(hashes=hashes,baseline_result_sha256=BASE_HASHES[seed],
                frozen_base_weights_sha256=digest,species=physics["species_names"],
                source_indices=sample["source_indices"].tolist(),
                resolved_components=int(sample["resolved"].sum()),
                excluded_active_components=int(len(x)*active.sum()-sample["resolved"].sum()),
                duplicate_feature_groups=int((counts>1).sum()),
                oracle=oracle(bv,sample["reference"],physics["species_names"],config,cuda=True),
                base=scores(bv,sample["reference"],sample["resolved"],bv,physics["species_names"]))
            save(root/"audit.json",audit)
            shared = {name:sha256(root/name) for name in ("sample.npz","preprocessing.npz","base-weights.pt","audit.json")}
            for width in WIDTHS:
                initial = None
                for objective in OBJECTIVES:
                    config = configuration(seed,width,objective)
                    destination = root/f"{width}-{objective}"
                    result = fit(x,bv,sample,active,physics["species_names"],config,destination)
                    if initial is not None and initial != result["initial_weights_sha256"]:
                        raise ValueError("Objective-paired initialization differs")
                    initial = result["initial_weights_sha256"]
                    if weight_hash(base) != digest:
                        raise ValueError("Frozen base changed")
                    result.update(status="complete",source=source,shared_artifacts=shared,
                        baseline_result_sha256=BASE_HASHES[seed],hashes=hashes,
                        artifacts={name:sha256(destination/name) for name in (
                            "weights.pt","optimizer.pt","predictions.npz","progress.jsonl","gradients.jsonl")})
                    save(destination/"result.json",result)
                    save(destination/"verification.json",dict(status="verified",result_sha256=sha256(destination/"result.json"),
                        exact_head_replay=True,cuda_numpy_pass_agreement=True,frozen_base=True,training_only=True))
                    status["completed"].append(dict(path=str(destination.relative_to(args.output)),result_sha256=sha256(destination/"result.json")))
                    save(args.output/"status.json",status)
        status["status"] = "complete"
        save(args.output/"status.json",status)
        save(args.output/"verification.json",dict(status="verified",status_sha256=sha256(args.output/"status.json"),
            environment_sha256=sha256(args.output/"environment.json"),paired_initializations=True,identical_training_subset=True,fit_count=8))
    except Exception as error:
        status.update(status="failed",error=str(error))
        save(args.output/"status.json",status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","baseline","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    plans = []
    for seed in SEEDS:
        _,_,hashes,sample,_ = load_inputs(args,seed)
        plans.append(dict(seed=seed,hashes=hashes,source_indices=sample["source_indices"].tolist(),
                          resolved_components=int(sample["resolved"].sum()),
                          fits=[configuration(seed,w,o) for w in WIDTHS for o in OBJECTIVES]))
    print(json.dumps(dict(status="planned",plan=plans)),flush=True)
    if args.execute:
        execute(args,plans)


if __name__ == "__main__":
    main()
