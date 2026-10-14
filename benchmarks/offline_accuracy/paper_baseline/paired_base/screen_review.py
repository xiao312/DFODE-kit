"""Independent full-population recount and fixed contract decision."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, verify_evaluator_lock
from benchmarks.research_contract.replay import inputs, population_rows, score
from benchmarks.research_contract.gates import comparison, decision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require, recount
from .metrics import align_indices


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--project",type=Path,required=True); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    project,root=args.project,args.run
    if (root/"review.json").exists(): raise ValueError("Review exists; do not overwrite")
    verify_evaluator_lock(project); binding=read_json(project/"research/bindings.json"); inputs(project,binding)
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,sha in report["artifacts"].items(): checked_file(root/path,sha)
    plan=read_json(root/"frozen-experiment.json"); policy=read_json(project/"research/campaign.json")["screening"]
    prepared,transfer=project/binding["prepared_root"],project/binding["transfer_root"]
    active=np.array([s!="AR" for s in binding["species"]]); records=[]; summary=[]; gates={}; latency=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed); other=next(s for s in plan["seeds"] if s!=seed)
        for model_seed in plan["seeds"]:
            for name in ("base-weights.pt","preprocessing.npz"):
                checked_file(staged/"model"/str(model_seed)/name,digest(prepared/str(model_seed)/name))
        for population,rows in population_rows(prepared,transfer,str(seed)):
            if population not in plan["populations"]: continue
            require(len(rows["states"])==plan["counts"][population],"Population count")
            for key,value in arrays(staged/(population+"-reference.npz")).items(): np.testing.assert_array_equal(value,rows[key])
            worker_input=arrays(staged/("input-"+population)/"states.npz")
            require(set(worker_input)=={"states","source_indices"},"Label isolation")
            for key,value in worker_input.items(): np.testing.assert_array_equal(value,rows[key])
            saved=arrays(root/str(seed)/population/"predictions.npz")
            np.testing.assert_array_equal(saved["source_indices"],rows["source_indices"])
            np.testing.assert_array_equal(saved["base"],rows["base"])
            # The other model cache uses the same named dataset namespace.
            other_rows=arrays(prepared/str(other)/("development.npz" if population=="development" else "training.npz"))
            available=np.isin(rows["source_indices"],other_rows["source_indices"])
            ix=align_indices(other_rows["source_indices"],rows["source_indices"][available])
            for key in ("states","delta","snapshot"): np.testing.assert_array_equal(other_rows[key][ix],rows[key][available])
            np.testing.assert_array_equal(saved["other"][available],other_rows["base"][ix]); del other_rows
            b,q,y=saved["base"],saved["other"],saved["prediction"]
            arithmetic={}
            if plan.get("backend")=="gpu-decode":
                old=arrays(project/plan["control_run"]/str(seed)/population/"predictions.npz")
                np.testing.assert_array_equal(saved["historical"],old["prediction"])
                np.testing.assert_array_equal(saved["source_indices"],old["source_indices"])
                budget=1e-15+.1*abs(rows["delta"])
                drift=abs(y-saved["historical"])/budget
                arithmetic=dict(max_budget_drift=float(drift[:,active].max()),
                    changed_pass_components=int(np.count_nonzero((abs(y-rows["delta"])<=budget)[:,active]!=(abs(saved["historical"]-rows["delta"])<=budget)[:,active])))
                require(arithmetic["max_budget_drift"]<=plan["maximum_budget_arithmetic_drift"],"Arithmetic drift exceeds preregistered implementation threshold")
                require(arithmetic["changed_pass_components"]==0,"Implementation changed pass counts")
                b,q=saved["backend_base"],saved["backend_other"]
            # Separate sign/min arithmetic, rather than calling candidate clipping.
            step=(q-b)/2; cap=plan["fraction"]*(1e-15+.1*abs(b))
            expected=b+np.sign(step)*np.minimum(abs(step),cap)
            if plan.get("mode")=="strict-shrink":
                expected=np.sign(b)*np.maximum((1-plan["fraction"])*abs(b),np.minimum(abs(b),np.sign(b)*expected))
                require(np.array_equal(np.sign(y),np.sign(b)),"Strict sign preservation")
                require(np.all(abs(y)<=abs(b)),"Magnitude preservation")
            np.testing.assert_array_equal(expected,y)
            np.testing.assert_array_equal(y[:,~active],b[:,~active])
            old_error=abs(b[:,active]-rows["delta"][:,active])/(1e-15+.1*abs(rows["delta"][:,active]))
            newpass=abs(y[:,active]-rows["delta"][:,active])<=1e-15+.1*abs(rows["delta"][:,active])
            safe=old_error<(1-plan["fraction"])/(1+.1*plan["fraction"])-1e-10
            require(np.all(newpass[safe]),"Preservation bound")
            repair=(old_error>1)&newpass
            require(not np.any(repair&(old_error>(1+plan["fraction"])/(1-.1*plan["fraction"])+1e-10)),"Repair-range bound")
            candidate=score(dict(prediction=y,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
            baseline=score(dict(prediction=rows["base"],source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
            recount(y,rows["delta"],rows["base"],candidate,active,rows["states"][:,2:])
            for snapshot in np.unique(rows["snapshot"]):
                mask=rows["snapshot"]==snapshot
                require(int(newpass[mask].sum())==candidate["slices"]["snapshot:"+str(snapshot)]["pass_count"],"Snapshot recount")
            result=read_json(root/str(seed)/population/"result.json")
            require(result["exact_reload_replay"],"Reload evidence missing")
            ratio=float(np.median(result["timings"]["candidate"])/np.median(result["timings"]["anchor"]))
            require(ratio==result["ratio"],"Timing ratio mismatch"); latency.append(ratio)
            if "optimized_single" in result["timings"]:
                arithmetic["latency_vs_optimized_single"]=float(np.median(result["timings"]["candidate"])/np.median(result["timings"]["optimized_single"]))
                arithmetic["optimized_single_vs_historical"]=float(np.median(result["timings"]["optimized_single"])/np.median(result["timings"]["anchor"]))
            for comparator in ("anchor","incumbent"): gates[f"{seed}:{population}:{comparator}"]=comparison(candidate,baseline,policy)
            records.append(dict(seed=seed,population=population,candidate=candidate,baseline=baseline))
            g=candidate["groups"]["all"]
            summary.append(dict(seed=seed,population=population,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],
                damage_rate=g["damage_rate_among_base_successes"],nontrivial=candidate["groups"]["nontrivial"]["acceptance"],p99=g["normalized_error"]["p99"],
                negative=candidate["endpoint_diagnostics"]["negative_components"],latency_ratio=ratio,
                other_cache_overlap_rows=int(available.sum()),other_cache_missing_rows=int((~available).sum()),
                physical_gate=gates[f"{seed}:{population}:anchor"],arithmetic=arithmetic))
    expected={f"{seed}:{population}:{comparator}" for seed in plan["seeds"] for population in plan["populations"] for comparator in ("anchor","incumbent")}
    verdict=decision(gates,expected,max(latency),policy)
    save(root/"protected-scores.json",dict(records=records)); verify_evaluator_lock(project)
    save(root/"review.json",dict(status="verified",summary=summary,decision=verdict,incumbent_unchanged=True,independent_pairs=1,
         limitations="One overlapping pair. Own anchor saw U; other seed prepared cache misses44/38 of its counterpart U rows, contrary to the pre-run all-seen shorthand. Fresh other-model reload covers ALL rows; cached parity applies only to overlap. No independent confirmation or promotion. Fixed bound repairs only near-threshold failures.",
         report_sha256=digest(root/"report.json"),scores_sha256=digest(root/"protected-scores.json")))
    print(summary); print(verdict)


if __name__=="__main__": main()
