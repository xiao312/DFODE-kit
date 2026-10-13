"""Independent teacher, hidden/head samples, full physical scores and cost checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf,expit
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score,population_rows
from benchmarks.research_contract.gates import comparison,decision
from ..hidden_features.prepare import save
from ..kinetic_features.review import require,recount
from ..paired_base.metrics import align_indices


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True);args=p.parse_args()
    project,root=args.project,args.run
    if (root/"review.json").exists():raise ValueError("Review exists")
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");inputs(project,binding)
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for name,sha in report["artifacts"].items():checked_file(root/name,sha)
    plan=read_json(root/"frozen-experiment.json");policy=read_json(project/"research/campaign.json")["screening"]
    prepared,transfer=project/binding["prepared_root"],project/binding["transfer_root"]
    selections=arrays(transfer/"plan/selections.npz");identities={};excluded=[]
    for seed in plan["seeds"]:
        with np.load(prepared/str(seed)/"training.npz") as z:ids=z["source_indices"]
        identities[seed]=ids;excluded.extend(ids[np.r_[np.arange(10000),selections[f"unseen_{seed}"]]])
    allowed=np.setdiff1d(np.intersect1d(*identities.values()),np.unique(excluded))
    ids=np.random.default_rng(plan["selection_seed"]).permutation(allowed)[:50000]
    active=np.array([s!="AR" for s in binding["species"]]);scores=[];summary=[];gates={a:{} for a in plan["arms"]};timing={a:[] for a in plan["arms"]};audits=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed);trained=root/str(seed);train=arrays(staged/"training-reference.npz")
        np.testing.assert_array_equal(train["source_indices"],ids)
        full=arrays(prepared/str(seed)/"training.npz");ix=align_indices(full["source_indices"],ids)
        for key,value in train.items():np.testing.assert_array_equal(value,full[key][ix])
        del full
        fitting=arrays(staged/"fit-input/training.npz");other=next(s for s in plan["seeds"] if s!=seed)
        full=arrays(prepared/str(other)/"training.npz");ix=align_indices(full["source_indices"],ids)
        for key in ("states","delta","snapshot"):np.testing.assert_array_equal(train[key],full[key][ix])
        np.testing.assert_array_equal(fitting["other"],full["base"][ix]);del full
        b,q=train["base"][:,active],fitting["other"][:,active]
        movement=np.minimum(.1*abs(b),1e-16+.01*abs(b));raw=np.divide(np.sign(b)*(b-q)/2,movement,out=np.zeros_like(b),where=movement>0)
        target=np.where(raw<0,0,np.where(raw>1,1,raw));saved_target=arrays(trained/"teacher.npz")["fraction"]
        np.testing.assert_allclose(target,saved_target,rtol=1e-13,atol=1e-15)
        constant=np.load(trained/"constant.npy");np.testing.assert_array_equal(constant,saved_target.mean(0))
        record=read_json(trained/"result.json");r=record["record"]
        require(r["updates_completed"]==6000 and r["row_presentations"]==60000000 and r["parameter_count"]==55034,"Work or model size differs")
        for name in ("base-weights.pt","preprocessing.npz"):checked_file(staged/"inference-model"/name,digest(prepared/str(seed)/name))
        for name in ("weights.pt","config.json","constant.npy"):checked_file(staged/"inference-model"/name,digest(trained/name))
        audit=arrays(trained/"hidden-audit.npz");indices=audit["indices"];prep=arrays(prepared/str(seed)/"preprocessing.npz")
        weights={k:v.double().numpy() for k,v in torch.load(prepared/str(seed)/"base-weights.pt",weights_only=True,map_location="cpu").items()}
        state=train["states"][indices];h=((np.column_stack([state[:,:2],state[:,2:]**.1])-prep["base_x_offset"])/prep["base_x_scale"]).astype(np.float32).astype(float)
        for layer in (0,2,4,6):
            v=h@weights[f"{layer}.weight"].T+weights[f"{layer}.bias"];h=.5*v*(1+erf(v/np.sqrt(2)))
        np.testing.assert_allclose(h,audit["features"],rtol=2e-4,atol=2e-4)
        hw={k:v.double().numpy() for k,v in torch.load(trained/"weights.pt",weights_only=True,map_location="cpu").items()}
        z=audit["features"].astype(float)@hw["0.weight"].T+hw["0.bias"]
        logits=(.5*z*(1+erf(z/np.sqrt(2))))@hw["2.weight"].T+hw["2.bias"]
        replay=arrays(trained/"training-replay.npz");np.testing.assert_allclose(expit(logits),replay["fraction"][indices],rtol=3e-5,atol=3e-6)
        audits.append(dict(seed=seed,teacher_max_budget_roundoff=record["teacher_formula_max_budget_difference"],teacher_pass_mask_differences=record["teacher_pass_mask_differences"],cpu_hidden_max_difference=float(abs(h-audit["features"]).max())))
        populations={"training":train}
        for population,rows in population_rows(prepared,transfer,str(seed)):
            if population!="training":populations[population]=rows
        for population,rows in populations.items():
            stateonly=arrays(staged/("input-"+population)/"states.npz");require(set(stateonly)=={"states","source_indices"},"Inference boundary")
            for key,value in stateonly.items():np.testing.assert_array_equal(value,rows[key])
            saved=arrays(trained/population/"predictions.npz");np.testing.assert_array_equal(saved["source_indices"],rows["source_indices"]);np.testing.assert_array_equal(saved["base"],rows["base"])
            if population=="training":
                np.testing.assert_array_equal(replay["prediction"],saved["head"]);np.testing.assert_array_equal(replay["fraction"],saved["head_fraction"])
            initial=rows["base"].copy();initial[:,active]-=.5*np.sign(initial[:,active])*.1*np.minimum(abs(initial[:,active]),1e-15+.1*abs(initial[:,active]))
            candidates={a:saved[a] for a in plan["arms"]};candidates["initial-half"]=initial
            measured=read_json(trained/population/"result.json")
            require(measured["exact_reload_replay"],"Reload missing")
            for arm,pred in candidates.items():
                if arm in plan["arms"]:
                    alpha=saved[arm+"_fraction"];require(np.all((alpha>=0)&(alpha<=1)),"Fraction range")
                    if arm=="constant":np.testing.assert_array_equal(alpha,np.broadcast_to(constant,alpha.shape))
                    expected=rows["base"].copy();bb=expected[:,active];m=.1*np.minimum(abs(bb),1e-15+.1*abs(bb))
                    v=bb-np.sign(bb)*m*alpha;expected[:,active]=np.sign(bb)*np.maximum(.9*abs(bb),np.minimum(abs(bb),np.sign(bb)*v))
                    np.testing.assert_array_equal(pred,expected)
                np.testing.assert_array_equal(np.sign(pred),np.sign(rows["base"]));require(np.all(abs(pred)<=abs(rows["base"])),"Magnitude growth")
                result=score(dict(prediction=pred,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
                baseline=score(dict(prediction=rows["base"],source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{population}")
                recount(pred,rows["delta"],rows["base"],result,active,rows["states"][:,2:]);g=result["groups"]["all"]
                ratio=next((t["ratio"] for t in measured["records"] if t["arm"]==arm),None)
                summary.append(dict(seed=seed,population=population,arm=arm,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],p99=g["normalized_error"]["p99"],negative=result["endpoint_diagnostics"]["negative_components"],latency_ratio=ratio))
                scores.append(dict(seed=seed,population=population,arm=arm,candidate=result,baseline=baseline))
                if population!="training" and arm in plan["arms"]:
                    for comparator in ("anchor","incumbent"):gates[arm][f"{seed}:{population}:{comparator}"]=comparison(result,baseline,policy)
                    timing[arm].append(ratio)
    keys={f"{s}:{p}:{c}" for s in plan["seeds"] for p in ("same_source_unseen","development") for c in ("anchor","incumbent")}
    verdict={a:decision(gates[a],keys,max(timing[a]),policy) for a in plan["arms"]}
    save(root/"protected-scores.json",dict(records=scores));verify_evaluator_lock(project)
    save(root/"review.json",dict(status="verified",summary=summary,decision=verdict,audits=audits,incumbent_unchanged=True,confirmation_opened=False,
         limits="Teacher imitation, oneoverlappingpair, no independentconfirmation; nearthresholdshrink only. CPUforward samples are implementation checks, not FP32 certification.",
         report_sha256=digest(root/"report.json"),scores_sha256=digest(root/"protected-scores.json")))
    print(summary);print(verdict)


if __name__=="__main__":main()
