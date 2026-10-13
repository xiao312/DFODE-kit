"""Independent formula recount of training-only finite-step effects."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,digest,checked_file
from ..hidden_features.prepare import save


def recount(truth,old,new):
    allowed=1e-15+np.abs(truth)/10
    before=np.less_equal(np.abs(old-truth),allowed);after=np.less_equal(np.abs(new-truth),allowed)
    nontrivial=np.abs(truth)>1e-15/.9
    masks=dict(all=np.ones_like(before),base_pass=before,base_fail=~before,floor=~nontrivial,
               weak_nontrivial=nontrivial&(np.abs(truth)<=1e-12),active_increment=np.abs(truth)>1e-12)
    result={}
    for name,mask in masks.items():
        n=int(mask.sum());oldcount=int(before[mask].sum());count=int(after[mask].sum())
        repair=int((after&~before&mask).sum());damage=int((before&~after&mask).sum())
        if count!=oldcount+repair-damage:raise ValueError("Repair identity failed")
        result[name]=dict(components=n,passes=count,repaired=repair,damaged=damage,
                         loss=float(np.log1p(np.abs(new-truth)/allowed)[mask].mean()) if n else None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path)
    args=p.parse_args();root=args.run
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    exceptions=[]
    for name,expected in report["artifacts"].items():
        actual=digest(root/name)
        if actual==expected:continue
        if name=="active-job.json" and read_json(root/name)==dict(status="COMPLETE_AWAITING_RECOUNT"):
            exceptions.append(dict(file=name,expected=expected,actual=actual,reason="a366ab6 updated mutable controller status after sealing; no scientific array changed"))
        else:raise ValueError("Artifact hash mismatch: "+name)
    summaries=[]
    for record in report["records"]:
        if record["status"]!="complete":
            summaries.append(dict(seed=record.get("seed"),status=record["status"],interpretation="No gradient conclusion permitted"));continue
        data=arrays(root/str(record["seed"])/"predictions.npz");batches=np.arange(len(data["truth"])).reshape(8,1024)
        rows=[]
        for item in record["records"]:
            index=item["batch"];effects={}
            for label,selection in [("step",batches[index]),("check",batches[(index+1)%8])]:
                before=recount(data["truth"][selection],data["baseline"][selection],data["baseline"][selection])
                after=recount(data["truth"][selection],data["baseline"][selection],data[f"prediction_{index}_{label}"])
                for phase,counts in [("before",before),("after",after)]:
                    for group,values in counts.items():
                        for key,value in values.items():
                            stored=item["populations"][label][phase][group][key]
                            if key=="loss" and value is not None:np.testing.assert_allclose(stored,value,rtol=1e-12,atol=1e-14)
                            elif stored!=value:raise ValueError("Count mismatch")
                effects[label]={group:dict(net_passes=after[group]["passes"]-before[group]["passes"],
                    repaired=after[group]["repaired"],damaged=after[group]["damaged"],
                    loss_change=None if before[group]["loss"] is None else after[group]["loss"]-before[group]["loss"]) for group in before}
            rows.append(dict(batch=index,cosine_pass_fail={key:value["cosines"][1][2] for key,value in item["geometry"].items()},
                             step_norm=item["step_norm"],first_order=item["first_order"],effects=effects))
        summaries.append(dict(seed=record["seed"],status="VERIFIED_TRAINING_DIAGNOSTIC",rows=rows,
            decoder_max=max(r["discrepancy"]["maximum"] for r in record["records"]),seconds=record["seconds"]))
    save(root/"review.json",dict(status="verified",report_sha256=digest(root/"report.json"),metadata_exceptions=exceptions,
         summaries=summaries,candidate_promoted=False,confirmation_opened=False,
         scope="Eight independent fresh-Adam steps per seed; 8192 training rows; not held-out model performance or a PCGrad benefit"))
    print(root/"review.json")


if __name__=="__main__":main()
