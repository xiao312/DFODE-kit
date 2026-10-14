"""Registered no-fit magnitude-preservation comparison on saved state-only outputs."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require,recount
from .metrics import shrink_only


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--execute",action="store_true");args=p.parse_args()
    project=args.project.resolve();plan=read_json(project/"research/experiments/EXP-20261013-consensus-shrink.json")
    root,output=project/plan["parent_run"],project/plan["output"]
    if output.exists():raise ValueError("Existing output")
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");inputs(project,binding)
    review=read_json(checked_file(root/"review.json",plan["review_sha256"]))
    report=read_json(checked_file(root/"report.json",review["report_sha256"]))
    for name,sha in report["artifacts"].items():checked_file(root/name,sha)
    if source_revision()["dirty"]:raise ValueError("Commit before execution")
    print("PRECHECKED",flush=True)
    if not args.execute:return
    output.mkdir();save(output/"frozen-experiment.json",plan);save(output/"source.json",source_revision())
    policy=read_json(project/"research/campaign.json")["screening"];active=np.array([s!="AR" for s in binding["species"]]);records=[];summary=[]
    for seed in plan["seeds"]:
        for population in plan["populations"]:
            rows=arrays(root/"prepared"/str(seed)/(population+"-reference.npz"));cached=arrays(root/str(seed)/population/"predictions.npz")
            np.testing.assert_array_equal(cached["source_indices"],rows["source_indices"]);np.testing.assert_array_equal(cached["base"],rows["base"])
            b,c,d=cached["base"],cached["prediction"],rows["delta"];pred=shrink_only(b,c)
            expected=np.where(b>=0,np.maximum(0,np.minimum(b,c)),np.minimum(0,np.maximum(b,c)))
            np.testing.assert_array_equal(pred,expected);require(np.all(abs(pred)<=abs(b)),"Magnitude grew")
            np.testing.assert_array_equal(pred[:,~active],b[:,~active])
            result=score(dict(prediction=pred,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
            baseline=score(dict(prediction=b,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
            recount(pred,d,b,result,active,rows["states"][:,2:])
            tau=1e-15+.1*abs(d[:,active]);old=abs(b[:,active]-d[:,active])<=tau
            before=abs(c[:,active]-d[:,active])<=tau;after=abs(pred[:,active]-d[:,active])<=tau
            g=result["groups"]["all"]
            summary.append(dict(seed=seed,population=population,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],
              negative=result["endpoint_diagnostics"]["negative_components"],p99=g["normalized_error"]["p99"],
              changed_from_bounded=int((pred[:,active]!=c[:,active]).sum()),lost_repairs=int((~old&before&~after).sum()),
              prevented_damage=int((old&~before&after).sum()),physical_gate=comparison(result,baseline,policy)))
            records.append(dict(seed=seed,population=population,candidate=result,baseline=baseline))
            np.savez(output/f"{seed}-{population}.npz",prediction=pred,source_indices=rows["source_indices"])
    save(output/"protected-scores.json",dict(records=records));verify_evaluator_lock(project)
    save(output/"review.json",dict(status="verified_cached_diagnostic",summary=summary,candidate=False,inference_latency=None,confirmation_opened=False,
         artifacts={p.name:digest(p) for p in output.iterdir() if p.is_file()},parent_review_sha256=plan["review_sha256"]))
    print(summary,flush=True)


if __name__=="__main__":main()
