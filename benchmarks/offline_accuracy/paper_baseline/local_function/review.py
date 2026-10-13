"""Second-pass raw-array checks independent of the protected scorer's formulas."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from benchmarks.research_contract.replay import population_rows
from ..hidden_features.prepare import save
from .audit_metrics import gate_failures,damage_slices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    args=p.parse_args()
    if (args.run/"review.json").exists():p.error("Review already exists")
    report=read_json(checked_file(args.run/"report.json",read_json(args.run/"verification.json")["report_sha256"]))
    for relative,expected in report["artifacts"].items():checked_file(args.run/relative,expected)
    binding=read_json(args.project/"research/bindings.json")
    active=np.array([s!="AR" for s in binding["species"]]);table=[];details=[];latencies={}
    policy=read_json(args.project/"research/campaign.json")["screening"]
    basecheck=read_json(args.run/"postcheck/report.json")
    if not basecheck["exact_all_populations"]:raise ValueError("Base replay missing")
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
                if int(passed.all(axis=1).sum())!=expected["all_active_species_state_pass_count"]:raise ValueError("State recount differs")
                if not np.isfinite(pred).all() or np.count_nonzero(pred[:,~active]):raise ValueError("Invalid returned predictions")
                masks=dict(all=np.ones_like(passed),nontrivial=abs(d)>1e-15/.9,
                           floor_dominated=abs(d)<=1e-15/.9,weak_nontrivial=(abs(d)>1e-15/.9)&(abs(d)<=1e-12),active_increment=abs(d)>1e-12)
                for name,mask in masks.items():
                    g=expected["groups"][name]
                    values=dict(components=int(mask.sum()),pass_count=int((passed&mask).sum()),
                                base_pass_count=int((old&mask).sum()),repaired=int((passed&~old&mask).sum()),damaged=int((~passed&old&mask).sum()))
                    if any(g[k]!=v for k,v in values.items()):raise ValueError("Group recount differs")
                    if mask.any():
                        np.testing.assert_allclose(np.quantile((abs(y-d)/tau)[mask],.99),g["normalized_error"]["p99"],rtol=1e-14,atol=0)
                for i,name in enumerate(np.array(binding["species"])[active]):
                    if int(passed[:,i].sum())!=expected["per_species"][name]["pass_count"]:raise ValueError("Species recount differs")
                nt=masks["nontrivial"]
                if int(((np.sign(y)!=np.sign(d))&nt).sum())!=expected["nontrivial_sign"]["wrong_sign_count"]:raise ValueError("Sign recount differs")
                for event in expected["small_target_events"]:
                    if int((abs(d)<=event["epsilon"]).sum())!=event["components"]:raise ValueError("Event denominator differs")
                    if int(((abs(d)<=event["epsilon"])&(abs(y)>=event["eta"])).sum())!=event["events"]:raise ValueError("Event recount differs")
                if int(((rows["states"][:,2:]+pred)<0).sum())!=expected["endpoint_diagnostics"]["negative_components"]:raise ValueError("Endpoint recount differs")
                g=expected["groups"]["all"]
                identity=old.mean()+(1-old.mean())*g["repair_rate_among_base_failures"]-old.mean()*g["damage_rate_among_base_successes"]
                if not np.isclose(passed.mean(),identity,rtol=0,atol=2e-16):raise ValueError("Repair identity differs")
                details.append(dict(seed=seed,arm=arm,population=population,
                                    **damage_slices(d,b,y,rows["snapshot"])))
                if population!="training":
                    failures=gate_failures(d,b,y,rows["states"][:,2:][:,active],policy,expected["small_target_events"],np.array(binding["species"])[active])
                    for comparator in ("anchor","incumbent"):
                        gate=record["gates"][f"{seed}:{population}:{comparator}"]
                        if failures!=sorted(gate["failures"]) or gate["missing"]:raise ValueError("Independent gates differ")
                    timing=read_json(args.run/"fits"/f"{seed}-{arm}"/population/"result.json")["timings"]
                    ratio=float(np.median(timing["candidate"])/np.median(timing["anchor"]))
                    latencies.setdefault(arm,[]).append(ratio)
                table.append(dict(seed=seed,arm=arm,population=population,states=len(d),acceptance=float(passed.mean()),
                    base=float(old.mean()),repaired=g["repaired"],damaged=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],
                    p99=float(np.quantile(abs(y-d)/tau,.99)),nontrivial=expected["groups"]["nontrivial"]["acceptance"],
                    weak=expected["groups"]["weak_nontrivial"]["acceptance"],active=expected["groups"]["active_increment"]["acceptance"],
                    sign_errors=expected["nontrivial_sign"]["wrong_sign_count"],negative=expected["endpoint_diagnostics"]["negative_components"]))
    for arm,ratios in latencies.items():
        failed=max(ratios)>policy["maximum_inference_latency_ratio"]
        if failed!=("inference" in report["decisions"][arm]["failures"]):raise ValueError("Latency gate differs")
    save(args.run/"review.json",dict(status="verified",report_sha256=digest(args.run/"report.json"),table=table,
         slices=details,latency_ratios=latencies,postcheck_sha256=digest(args.run/"postcheck/report.json"),
         decisions=report["decisions"],incumbent_changed=False,confirmation_opened=False))
    print(args.run/"review.json")


if __name__=="__main__":main()
