"""Check pinned baselines and execute one isolated residual-coordinate experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.improve.coordinates import transitions
from ..matched_work.run import inputs,checked_result
from ..matched_targets.model import reload_model as load_base,prediction as base_prediction
from ..run import evaluate_and_verify
from .plan import configuration,base_config,ARMS,SEEDS,BASE_HASHES,CAMPAIGN_HASH
from .coordinates import preprocessing
from .model import prediction,reload_model
from .fit import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign","previous","original","base-root","baseline","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--arm",choices=ARMS,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    config = configuration(args.arm,args.seed)
    if sha256(args.baseline/"verification.json") != CAMPAIGN_HASH:
        raise ValueError("Frozen campaign identity differs")
    base_path = args.baseline/f"{args.seed}-gbct-increment"
    if sha256(base_path/"result.json") != BASE_HASHES[args.seed]:
        raise ValueError("Pinned GBCT baseline changed")
    baseline = checked_result(base_path)
    if baseline["config"] != base_config(args.seed):
        raise ValueError("Baseline configuration differs")
    training,normalization,development,physics,audit,hashes = inputs(
        args.campaign,args.previous,args.original,args.base_root,"fuel-state",args.seed)
    if (len(training["states"]),len(normalization["states"]),len(development["states"]),
            len(physics["species_names"]),physics["interval"]) != (200000,50000,1023,59,config["interval"]):
        raise ValueError("Dataset size, mechanism or interval differs from the frozen plan")
    if baseline["hashes"] != hashes:
        raise ValueError("Training data identity differs")
    for name,rows in (("training-indices.npy",training),("normalization-indices.npy",normalization)):
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
        base,base_prep = load_base(base_path,base_config(args.seed))
        models,prep,fitted = fit(base,base_prep,training,development,normalization,config,args.output)
        result.update(fitted)
        def rebuild(rows,species,conf):
            values,_ = base_prediction(base,base_prep,rows["states"],base_config(args.seed))
            return preprocessing(base_prep,values,rows["delta"],conf)
        result.update(evaluate_and_verify(models,prep,training,development,physics,audit,config,args.output,
            args.campaign/"dataset",normalization_training=normalization,
            preprocess_fn=rebuild,predict_fn=prediction,reload_fn=reload_model))
        for split,rows in (("training",training),("development",development)):
            with np.load(base_path/f"{split}-predictions.npz") as old, np.load(args.output/f"{split}-predictions.npz") as new:
                np.testing.assert_array_equal(old["source_indices"],new["source_indices"])
                result[split+"_transitions"] = transitions(old["prediction"],new["prediction"],rows["delta"],physics["species_names"])
        result["artifacts"] = {name:sha256(args.output/name) for name in (
            "environment.json","base-weights.pt","weights.pt","optimizer.pt","preprocessing.npz",
            "training-indices.npy","normalization-indices.npy","training-predictions.npz","development-predictions.npz","progress.jsonl")}
        result["status"] = "complete"
        save(args.output/"result.json",result)
        save(args.output/"verification.json",dict(status="verified",result_sha256=sha256(args.output/"result.json"),
            exact_model_replay=True,independent_paired_counts=True,physical_checks=True,train_only_calibration=True,frozen_base=True))
    except Exception as error:
        result.update(status="failed",error=str(error))
        save(args.output/"result.json",result)
        raise


if __name__ == "__main__":
    main()
