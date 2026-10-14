"""Pinned baseline gate, no-training audit, matched fit and independent replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.evaluate import audit_subset
from benchmarks.offline_accuracy.improve.coordinates import transitions
from ..matched_work.run import inputs,checked_result
from ..matched_targets.model import reload_model as load_base,prediction as base_prediction
from ..adaptive.coordinates import preprocessing
from ..run import evaluate_and_verify
from .plan import configuration,base_config,ARMS,SEEDS,BASE_HASHES,CAMPAIGN_HASH
from .model import prediction,reload_model
from .fit import fit
from .diagnostics import oracle,tails


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","baseline","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    config = configuration(args.arm,args.seed)
    if sha256(args.baseline/"verification.json")!=CAMPAIGN_HASH:
        raise ValueError("Frozen baseline campaign changed")
    base_path = args.baseline/f"{args.seed}-gbct-increment"
    if sha256(base_path/"result.json")!=BASE_HASHES[args.seed]:
        raise ValueError("Frozen base result changed")
    baseline = checked_result(base_path)
    if baseline["config"]!=base_config(args.seed):
        raise ValueError("Baseline configuration differs")
    training,norm,development,physics,audit,hashes = inputs(
        args.campaign,args.previous,args.original,args.base_root,"fuel-state",args.seed)
    if (len(training["states"]),len(norm["states"]),len(development["states"]),len(physics["species_names"]),
            physics["interval"])!=(200000,50000,1023,59,config["interval"]):
        raise ValueError("Frozen dataset dimensions/interval differ")
    if baseline["hashes"]!=hashes:
        raise ValueError("Data hashes differ")
    for name,rows in (("training-indices.npy",training),("normalization-indices.npy",norm)):
        np.testing.assert_array_equal(np.load(base_path/name),rows["source_indices"])
    result = dict(status="planned",config=config,hashes=hashes,baseline_result_sha256=BASE_HASHES[args.seed],
        training_count=len(training["states"]),development_count=len(development["states"]),independent_test_count=0)
    print(json.dumps(result),flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit scientific source before execution")
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    result.update(status="running",source=source)
    save(args.output/"result.json",result)
    try:
        base,bp = load_base(base_path,base_config(args.seed))
        bases,audits = {},{}
        for split,rows in (("training",training),("development",development)):
            values,_ = base_prediction(base,bp,rows["states"],base_config(args.seed))
            with np.load(base_path/f"{split}-predictions.npz") as old:
                np.testing.assert_array_equal(old["source_indices"],rows["source_indices"])
                np.testing.assert_array_equal(old["prediction"],values)
            bases[split] = values
            audits[split] = {decoder:oracle(values,rows["delta"],physics["species_names"],dict(config,decoder=decoder),cuda=True)
                             for decoder in ("linear","asinh")}
        save(args.output/"decoder-audit.json",audits)
        if args.audit_only:
            result.update(status="audit-complete",decoder_audit_sha256=sha256(args.output/"decoder-audit.json"))
            save(args.output/"result.json",result)
            return
        norm_values,_ = base_prediction(base,bp,norm["states"],base_config(args.seed))
        prep = preprocessing(bp,norm_values,norm["delta"],config)
        models,fitted = fit(base,prep,bases["training"],training,development,norm,config,args.output)
        result.update(fitted)
        def rebuild(rows,species,conf):
            values,_ = base_prediction(base,bp,rows["states"],base_config(args.seed))
            return preprocessing(bp,values,rows["delta"],conf)
        result.update(evaluate_and_verify(models,prep,training,development,physics,audit,config,args.output,
            args.campaign/"dataset",normalization_training=norm,
            preprocess_fn=rebuild,predict_fn=prediction,reload_fn=reload_model))
        detail = {}
        for split,rows in (("training",training),("development",development)):
            with np.load(args.output/f"{split}-predictions.npz") as new:
                predicted = new["prediction"]
                result[split+"_transitions"] = transitions(bases[split],predicted,rows["delta"],physics["species_names"])
                detail[split] = dict(base=tails(bases[split],rows["delta"],physics["species_names"]),
                    fitted=tails(predicted,rows["delta"],physics["species_names"]))
                if split=="development":
                    indices,reference,uncertainty = audit_subset(rows,audit)
                    detail["audited_development"] = dict(
                        base=tails(bases[split][indices],reference,physics["species_names"],uncertainty),
                        fitted=tails(predicted[indices],reference,physics["species_names"],uncertainty))
        save(args.output/"increment-diagnostics.json",detail)
        result["artifacts"] = {name:sha256(args.output/name) for name in (
            "environment.json","base-weights.pt","weights.pt","optimizer.pt","preprocessing.npz",
            "training-indices.npy","normalization-indices.npy","training-predictions.npz","development-predictions.npz",
            "progress.jsonl","gradients.jsonl","decoder-audit.json","increment-diagnostics.json")}
        result["status"] = "complete"
        save(args.output/"result.json",result)
        save(args.output/"verification.json",dict(status="verified",result_sha256=sha256(args.output/"result.json"),
            exact_model_replay=True,independent_paired_counts=True,physical_checks=True,train_only_preprocessing=True,
            frozen_base=True,baseline_prediction_replay=True))
    except Exception as error:
        result.update(status="failed",error=str(error))
        save(args.output/"result.json",result)
        raise


if __name__=="__main__":
    main()
