"""Registered descriptive cross-tab, not a deployable oracle selector."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);args=p.parse_args()
    project=args.project;plan=read_json(project/"research/experiments/EXP-20261013-rate-overlap.json")
    root=project/plan["parent_run"];output=project/plan["output"]
    if output.exists():raise ValueError("Existing diagnostic; do not overwrite")
    verify_evaluator_lock(project);review=read_json(checked_file(root/"review.json",plan["review_sha256"]))
    report=read_json(checked_file(root/"report.json",review["report_sha256"]))
    binding=read_json(project/"research/bindings.json");active=np.array([s!="AR" for s in binding["species"]]);records=[];summary=[]
    for seed in plan["seeds"]:
        original=f"prepared/{seed}/training.npz";predictions=f"{seed}/inference/predictions.npz"
        rows=arrays(checked_file(root/original,report["artifacts"][original]));saved=arrays(checked_file(root/predictions,report["artifacts"][predictions]))
        np.testing.assert_array_equal(rows["source_indices"],saved["source_indices"])
        for population,sl in dict(fitting20k=slice(0,20000),check20k=slice(30000,None)).items():
            d=rows["delta"][sl][:,active];base=rows["base"][sl][:,active];tau=1e-15+.1*abs(d)
            old=abs(base-d)<=tau;raw=saved["rate_raw"][sl];rawpass=(abs(raw[:,:58]-d)<=tau)|(abs(raw[:,58:]-d)<=tau)
            nt=abs(d)>1e-15/.9;groups=dict(floor=~nt,weak=nt&(abs(d)<=1e-12),active=abs(d)>1e-12)
            for group,mask in groups.items():
                for base_pass in (False,True):
                    for rate_pass in (False,True):
                        subset=mask&(old==base_pass)&(rawpass==rate_pass)
                        record=dict(seed=seed,population=population,group=group,base_pass=base_pass,rate_pass=rate_pass,components=int(subset.sum()),states=int(subset.any(1).sum()),arms={})
                        for arm in ("ordinary","kinetic","capacity"):
                            new=abs(saved[arm][sl][:,active]-d)<=tau
                            record["arms"][arm]=dict(passes=int((new&subset).sum()),repairs=int((new&~old&subset).sum()),damage=int((~new&old&subset).sum()))
                        records.append(record)
            eligible=~old&rawpass;kinetic=abs(saved["kinetic"][sl][:,active]-d)<=tau
            summary.append(dict(seed=seed,population=population,base_failures=int((~old).sum()),raw_repair_opportunities=int(eligible.sum()),kinetic_recovers=int((eligible&kinetic).sum()),kinetic_repairs_without_raw_pass=int((~old&~rawpass&kinetic).sum()),oracle_union_acceptance=float((old|rawpass).mean())))
    save(output,dict(status="verified_descriptive",plan=plan,review_sha256=plan["review_sha256"],summary=summary,records=records,candidate=False))
    print(output);print(summary)


if __name__=="__main__":main()
