"""Execute a regression gate or one fixed-work rung against immutable prepared inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.paired.runtime import configure
from ..matched_work.run import checked_result
from ..resolved_fit.run import require_idle
from ..resolved_fit.fit import fit as probe_fit
from ..resolved_fit.plan import configuration as probe_configuration
from ..decoder_loss.diagnostics import tails
from .plan import SEEDS,SIZES,OBJECTIVES,CONTINUED_HASHES,configuration
from .model import load_base,load_head,prediction,cached_prediction
from .evaluate import summary,audit_scores
from .train import fit


def prepared_manifest(root,seed):
    status = json.loads((root/"status.json").read_text())
    verification = json.loads((root/"verification.json").read_text())
    if status["status"]!="complete" or verification["status_sha256"]!=sha256(root/"status.json"):
        raise ValueError("Prepared campaign not complete or changed")
    entry = next(r for r in status["seeds"] if r["seed"]==seed)
    directory = root/str(seed)
    if sha256(directory/"manifest.json")!=entry["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    m = json.loads((directory/"manifest.json").read_text())
    if m["continued_result_sha256"]!=CONTINUED_HASHES[seed]:
        raise ValueError("Incumbent identity differs")
    for name,digest in m["artifacts"].items():
        if Path(name).name!=name or sha256(directory/name)!=digest:
            raise ValueError("Prepared artifact changed: "+name)
    return m


def check_probes(root,prepared):
    if root is None:
        raise ValueError("Supply --probe-root containing both verified continued-base regression fits")
    for seed in SEEDS:
        r = checked_result(root/str(seed)/"fit")
        if r["prepared_manifest_sha256"]!=sha256(prepared/str(seed)/"manifest.json"):
            raise ValueError("Probe used different prepared base or data")
        if any(r["metrics"][k]["acceptance"]<.99 for k in ("resolved","nontrivial")):
            raise ValueError("Continued-base small-set regression gate failed")


def load_arrays(path):
    with np.load(path,allow_pickle=False) as a:
        return dict(a)


def execute(args,manifest):
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit scientific source first")
    require_idle()
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    root = args.prepared/str(args.seed)
    with np.load(root/"preprocessing.npz") as a:
        prep = dict(a)
    active = prep["active"]
    species = manifest["species"]
    common = dict(source=source,continued_result_sha256=CONTINUED_HASHES[args.seed],
                  prepared_manifest_sha256=sha256(root/"manifest.json"),independent_test_count=0)
    result = dict(common,status="running")
    save(args.output/"result.json",result)
    try:
        if args.probe:
            sample = load_arrays(root/"probe.npz")
            cfg = probe_configuration(args.seed,256,"direct")
            fitted = probe_fit(sample["features"],sample["base"],sample,active,species,cfg,args.output/"fit")
            fitted.update(common,status="complete",artifacts={p.name:sha256(p) for p in (args.output/"fit").iterdir()})
            save(args.output/"fit/result.json",fitted)
            save(args.output/"fit/verification.json",dict(status="verified",result_sha256=sha256(args.output/"fit/result.json")))
            result.update(status="complete",probe=fitted["metrics"],fit_result_sha256=sha256(args.output/"fit/result.json"))
        else:
            cfg = configuration(args.seed,args.size,args.objective)
            training = {k:v[:args.size] for k,v in load_arrays(root/"training.npz").items()}
            development = load_arrays(root/"development.npz")
            model,fitted = fit(training,development,active,cfg,args.output)
            result.update(fitted,config=cfg,training_count=args.size,development_count=len(development["states"]))
            base,_ = load_base(root,args.seed)
            restored = load_head(args.output,prep,cfg)
            audit = json.loads((root/"audit.json").read_text())
            details = {}
            for split,rows,audit_split in (("training",training,"train"),("development",development,"validation")):
                p,c = cached_prediction(model,rows,active,cfg)
                replay,rc = prediction(base,restored,prep,rows["states"],cfg)
                np.testing.assert_array_equal(c,rc)
                np.testing.assert_array_equal(p,replay)
                details[split] = summary(p,c,rows,rows["base"],species,cfg)
                result[split] = {k:v for k,v in details[split].items() if k not in ("groups","species","magnitude_bins")}
                result[split+"_base"] = tails(rows["base"],rows["delta"],species)
                result[split+"_audited"] = audit_scores(p,rows,audit,species,audit_split)
                np.savez(args.output/f"{split}-predictions.npz",prediction=p,coordinate=c,source_indices=rows["source_indices"])
            save(args.output/"group-scores.json",details)
            timings = []
            for _ in range(5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction(base,model,prep,development["states"],cfg)
                torch.cuda.synchronize()
                timings.append(time.perf_counter()-started)
            result.update(status="complete",inference=dict(states=len(development["states"]),seconds=timings),
                artifacts={p.name:sha256(p) for p in args.output.iterdir() if p.is_file() and p.name!="result.json"})
        save(args.output/"result.json",result)
        save(args.output/"verification.json",dict(status="verified",result_sha256=sha256(args.output/"result.json"),
            exact_replay=True,unchanged_prepared_manifest=sha256(root/"manifest.json")))
    except Exception as error:
        result.update(status="failed",error=str(error))
        save(args.output/"result.json",result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    parser.add_argument("--size",type=int,choices=SIZES,default=10000)
    parser.add_argument("--objective",choices=OBJECTIVES,default="direct")
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--probe-root",type=Path)
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new result directory")
    manifest = prepared_manifest(args.prepared,args.seed)
    if not args.probe:
        check_probes(args.probe_root,args.prepared)
    elif args.objective!="direct":
        parser.error("The regression gate uses direct supervision only")
    print(json.dumps(dict(status="planned",probe=args.probe,config=configuration(args.seed,args.size,args.objective))),flush=True)
    if args.execute:
        execute(args,manifest)


if __name__=="__main__":
    main()
