"""Second-pass raw-array checks independent of the protected scorer's formulas."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    if (args.run/"review.json").exists():p.error("Review already exists")
    report=read_json(checked_file(args.run/"report.json",read_json(args.run/"verification.json")["report_sha256"]))
    for relative,expected in report["artifacts"].items():checked_file(args.run/relative,expected)
    binding=read_json(args.project/"research/bindings.json")
    active=np.array([s!="AR" for s in binding["species"]]);table=[]
    plan=read_json(args.run/"frozen-experiment.json")
    for seed in plan["seeds"]:
        centres=[]
        for arm in plan["arms"]:
            root=args.run/"fits"/f"{seed}-{arm}"
            fit=read_json(root/"fit/result.json")
            if fit["parameter_count"]!=plan["parameter_count"] or fit["history"][-1]["update"]!=plan["updates"]:
                raise ValueError("Work or parameter count differs")
            geom=arrays(root/"model/geometry.npz");centres.append(geom["centre_ids"])
        if len(centres)>1:np.testing.assert_array_equal(centres[0],centres[1])
        for population,rows in population_rows(args.project/binding["prepared_root"],args.project/binding["transfer_root"],str(seed)):
            if population=="training":
                ix=np.load(args.run/"prepared"/str(seed)/"training-indices.npy",allow_pickle=False)
                full=arrays(args.project/binding["prepared_root"]/str(seed)/"training.npz")
                rows={k:v[ix] for k,v in full.items()};del full
            d=rows["delta"][:,active];b=rows["base"][:,active];tau=1e-15+.1*abs(d);old=abs(b-d)<=tau
            for arm in plan["arms"]:
                record=next(r for r in report["records"] if r["seed"]==seed and r["arm"]==arm)
                expected=record["populations"][population]["candidate"]
                saved=arrays(args.run/"fits"/f"{seed}-{arm}"/population/"predictions.npz")
                np.testing.assert_array_equal(saved["source_indices"],rows["source_indices"])
                pred=saved["prediction"];y=pred[:,active];passed=abs(y-d)<=tau
                if not np.isfinite(pred).all() or np.count_nonzero(pred[:,~active]):raise ValueError("Invalid returned predictions")
                masks=dict(all=np.ones_like(passed),nontrivial=abs(d)>1e-15/.9,
                           floor_dominated=abs(d)<=1e-15/.9,weak_nontrivial=(abs(d)>1e-15/.9)&(abs(d)<=1e-12),active_increment=abs(d)>1e-12)
                for name,mask in masks.items():
                    g=expected["groups"][name]
                    values=dict(components=int(mask.sum()),pass_count=int((passed&mask).sum()),
                                base_pass_count=int((old&mask).sum()),repaired=int((passed&~old&mask).sum()),damaged=int((~passed&old&mask).sum()))
                    if any(g[k]!=v for k,v in values.items()):raise ValueError("Group recount differs")
                for i,name in enumerate(np.array(binding["species"])[active]):
                    if int(passed[:,i].sum())!=expected["per_species"][name]["pass_count"]:raise ValueError("Species recount differs")
                nt=masks["nontrivial"]
                if int(((np.sign(y)!=np.sign(d))&nt).sum())!=expected["nontrivial_sign"]["wrong_sign_count"]:raise ValueError("Sign recount differs")
                for event in expected["small_target_events"]:
                    if int(((abs(d)<=event["epsilon"])&(abs(y)>=event["eta"])).sum())!=event["events"]:raise ValueError("Event recount differs")
                if int(((rows["states"][:,2:]+pred)<0).sum())!=expected["endpoint_diagnostics"]["negative_components"]:raise ValueError("Endpoint recount differs")
                g=expected["groups"]["all"]
                identity=old.mean()+(1-old.mean())*g["repair_rate_among_base_failures"]-old.mean()*g["damage_rate_among_base_successes"]
                if not np.isclose(passed.mean(),identity,rtol=0,atol=2e-16):raise ValueError("Repair identity differs")
                table.append(dict(seed=seed,arm=arm,population=population,states=len(d),acceptance=float(passed.mean()),
                    base=float(old.mean()),repaired=g["repaired"],damaged=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],
                    p99=float(np.quantile(abs(y-d)/tau,.99)),nontrivial=expected["groups"]["nontrivial"]["acceptance"],
                    weak=expected["groups"]["weak_nontrivial"]["acceptance"],active=expected["groups"]["active_increment"]["acceptance"],
                    sign_errors=expected["nontrivial_sign"]["wrong_sign_count"],negative=expected["endpoint_diagnostics"]["negative_components"]))
    save(args.run/"review.json",dict(status="verified",report_sha256=digest(args.run/"report.json"),table=table,
         decisions=report["decisions"],incumbent_changed=False,confirmation_opened=False))
    print(args.run/"review.json")


if __name__=="__main__":main()
