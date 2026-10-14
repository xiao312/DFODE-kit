"""GPU inference only: fixed damping, same-source unseen states and damage margins."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.paired.runtime import configure
from ..resolved_fit.run import require_idle
from ..correction_scaling.run import load_arrays
from ..correction_scaling.model import load_base,load_head,prediction,cached_prediction
from ..correction_scaling.plan import SEEDS,configuration
from ..correction_scaling.coverage import group_scores
from ..decoder_loss.coordinates import decode
from .prepare import inputs,checked_plan
from .metrics import score,preservation


def evaluate_seed(args,plan,seed):
    prepared = args.original/"prepared"/str(seed)
    cfg = configuration(seed,10000,"coordinate-sl1")
    base,prep = load_base(prepared,seed)
    model = load_head(args.candidate/"10000"/f"{seed}-coordinate-sl1",prep,cfg)
    active = prep["active"]
    full = load_arrays(prepared/"training.npz")
    indices = load_arrays(args.plan/"selections.npz")[f"unseen_{seed}"]
    if len(indices)!=20000 or len(np.unique(indices))!=20000 or np.min(indices)<10000:
        raise ValueError("Invalid correction-unseen selection")
    populations = dict(training={k:v[:10000] for k,v in full.items()},
        same_source_unseen={k:v[indices] for k,v in full.items()},
        development=load_arrays(prepared/"development.npz"))
    report = dict(seed=seed,populations={})
    for name,data in populations.items():
        p,c = cached_prediction(model,data,active,cfg)
        replay,rc = prediction(base,model,prep,data["states"],cfg)
        np.testing.assert_array_equal(p,replay); np.testing.assert_array_equal(c,rc)
        if name!="same_source_unseen":
            saved = load_arrays(args.candidate/"10000"/f"{seed}-coordinate-sl1"/f"{name}-predictions.npz")
            np.testing.assert_array_equal(saved["prediction"],p)
            np.testing.assert_array_equal(saved["coordinate"],c)
            np.testing.assert_array_equal(saved["source_indices"],data["source_indices"])
        curves = []
        for gamma in plan["gammas"]:
            predicted = data["base"].copy()
            predicted[:,active] = decode(data["base"][:,active],gamma*c,cfg)
            if gamma==0:
                np.testing.assert_array_equal(predicted,data["base"])
            record = dict(gamma=gamma,**score(predicted[:,active],data["delta"][:,active],data["base"][:,active]))
            record["snapshots"] = []
            for snapshot in np.unique(data["snapshot"]):
                mask = data["snapshot"]==snapshot
                record["snapshots"].append(dict(snapshot=str(snapshot),states=int(mask.sum()),
                    **score(predicted[mask][:,active],data["delta"][mask][:,active],data["base"][mask][:,active])))
            if gamma in (0,1):
                record["groups"] = group_scores(predicted,data,plan["species"])
            curves.append(record)
        result = dict(states=len(p),exact_label_free_replay=True,curves=curves,
            preservation=preservation(data["base"][:,active],data["delta"][:,active],c,cfg))
        if name=="same_source_unseen":
            result["feature_calibration_seen_states"] = int((indices<50000).sum())
        report["populations"][name] = result
        np.savez(args.output/f"{seed}-{name}.npz",coordinate=c,prediction=p,source_indices=data["source_indices"])
        print(json.dumps(dict(seed=seed,population=name,base=curves[0]["acceptance"],full=curves[-1]["acceptance"],
            damage=curves[-1]["damage_rate"],repairs=curves[-1]["repair_rate"])),flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan","original","candidate","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new evaluation directory")
    plan = checked_plan(args.plan)
    inputs(args.original,args.candidate)
    for seed in SEEDS:
        if sha256(args.original/"prepared"/str(seed)/"manifest.json")!=plan["prepared_hashes"][str(seed)]:
            raise ValueError("Plan prepared identity differs")
    print(json.dumps(dict(status="planned",gammas=plan["gammas"],no_training=True)),flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit evaluation source first")
    require_idle()
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    report = dict(source=source,plan_sha256=sha256(args.plan/"plan.json"),seeds=[])
    for seed in SEEDS:
        report["seeds"].append(evaluate_seed(args,plan,seed))
    report.update(status="complete",artifacts={q.name:sha256(q) for q in args.output.iterdir()})
    save(args.output/"report.json",report)
    save(args.output/"verification.json",dict(report_sha256=sha256(args.output/"report.json")))


if __name__=="__main__":
    main()
