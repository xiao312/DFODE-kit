"""Replay the frozen base from isolated, label-free population inputs."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,digest,checked_file,verify_evaluator_lock
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save

POPULATIONS=("training","same_source_unseen","development")


def worker(seed):
    from benchmarks.offline_accuracy.paired.runtime import configure
    from ..correction_scaling.model import load_base,base_prediction
    from ..adaptive.model import base_preprocessing
    from ..decoder_loss.coordinates import decode
    configure();base,prep=load_base(Path("/model"),seed)
    for population in POPULATIONS:
        rows=arrays(Path("/input")/(population+".npz"))
        if set(rows)!={"states","source_indices"}:raise ValueError("Labels in replay input")
        value,_=base_prediction(base,base_preprocessing(prep),rows["states"],seed)
        np.testing.assert_array_equal(value,decode(value,np.zeros_like(value),dict(decoder="asinh",scale_floor=1e-14)))
        np.savez(Path("/output")/(population+".npz"),prediction=value,source_indices=rows["source_indices"])
    save(Path("/output/result.json"),dict(status="complete",zero_decoder_exact=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path);p.add_argument("--run",type=Path)
    p.add_argument("--worker",action="store_true");p.add_argument("--seed",type=int)
    args=p.parse_args()
    if args.worker:return worker(args.seed)
    from ..hidden_features.campaign import run_worker
    verify_evaluator_lock(args.project)
    checked_file(args.run/"report.json",read_json(args.run/"verification.json")["report_sha256"])
    destination=args.run/"postcheck"
    if destination.exists():raise ValueError("Postcheck exists: reconcile before any retry")
    destination.mkdir();binding=read_json(args.project/"research/bindings.json")
    plan=read_json(args.run/"frozen-experiment.json")
    for seed in plan["seeds"]:
        source=args.run/"prepared"/str(seed);inputs=destination/str(seed)/"input";inputs.mkdir(parents=True)
        model=destination/str(seed)/"model";model.mkdir()
        for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(source/"model"/name,model/name)
        for population in POPULATIONS:
            shutil.copyfile(source/("input-"+population)/"states.npz",inputs/(population+".npz"))
        output=destination/str(seed)/"output"
        run_worker(args.project,inputs,output,["--worker","--seed",str(seed)],destination/"active-job.json",0,model,
                   worker_module="benchmarks.offline_accuracy.paper_baseline.local_function.postcheck")
        for population,rows in population_rows(args.project/binding["prepared_root"],args.project/binding["transfer_root"],str(seed)):
            if population=="training":
                ix=np.load(source/"training-indices.npy",allow_pickle=False)
                full=arrays(args.project/binding["prepared_root"]/str(seed)/"training.npz")
                rows={k:v[ix] for k,v in full.items()};del full
            actual=arrays(output/(population+".npz"))
            np.testing.assert_array_equal(actual["source_indices"],rows["source_indices"])
            np.testing.assert_array_equal(actual["prediction"],rows["base"])
    save(destination/"report.json",dict(exact_all_populations=True,seeds=plan["seeds"],
         artifacts={str(p.relative_to(destination)):digest(p) for p in destination.rglob("*") if p.is_file()}))
    print("Exact frozen-base and zero-correction replay on T50k, U20k and D1023, both seeds")


if __name__=="__main__":main()
