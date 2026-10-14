"""Trusted physical scoring and separate selector/split/count reconstruction."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import erf
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import score
from ..hidden_features.prepare import save


def require(condition,message):
    if not condition:raise ValueError(message)


def choose_independently(prob,b,c,initial,d):
    tau=1e-15+.1*abs(d);basepass=abs(b-d)<=tau;candidatepass=abs(c-d)<=tau;nt=abs(d)>1e-15/.9
    options=[]
    for threshold in list(np.arange(101)/100)+[1.01]:
        mask=(prob>=threshold)&(initial+c>=0)
        p=np.where(mask,c,b);passed=abs(p-d)<=tau
        repairs=int((passed&~basepass).sum());damage=int((~passed&basepass).sum())
        valid=(damage<=basepass.sum()/100 and passed[nt].sum()>=basepass[nt].sum() and ((np.sign(p)!=np.sign(d))&nt).sum()<=((np.sign(b)!=np.sign(d))&nt).sum())
        if valid:options.append((repairs-damage,-int(mask.sum()),float(threshold)))
    return max(options)[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True);args=p.parse_args();root=args.run
    if (root/"review.json").exists():p.error("Review exists")
    verify_evaluator_lock(args.project);binding=read_json(args.project/"research/bindings.json")
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,sha in report["artifacts"].items():checked_file(root/path,sha)
    plan=read_json(root/"frozen-experiment.json");active=np.array([name!="AR" for name in binding["species"]]);table=[];scores=[];passes=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed);original=arrays(staged/"training.npz");rows=arrays(root/str(seed)/"predictions.npz");result=read_json(root/str(seed)/"result.json")
        full=arrays(args.project/binding["prepared_root"]/str(seed)/"training.npz")
        unseen=arrays(args.project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
        allowed=np.setdiff1d(np.arange(len(full["states"])),np.r_[np.arange(10000),unseen])
        indices=np.random.default_rng(seed+701).permutation(allowed)[:50000]
        np.testing.assert_array_equal(indices,np.load(staged/"selected-positions.npy"))
        require(not np.isin(indices,unseen).any() and np.min(indices)>=10000,"Role exclusion failed")
        for key,value in original.items():np.testing.assert_array_equal(value,full[key][indices])
        del full
        np.testing.assert_array_equal(rows["source_indices"],original["source_indices"])
        np.testing.assert_array_equal(result["active_species_indices"],np.flatnonzero(active))
        for key,value in dict(truth=original["delta"][:,active],baseline=original["base"][:,active],initial=original["states"][:,2:][:,active],features=original["features"]).items():np.testing.assert_array_equal(rows[key],value)
        scale=np.maximum(np.sqrt((rows["coordinate"][:20000]**2).mean(0)),1e-6);np.testing.assert_array_equal(scale,rows["scale"])
        d,b,c,initial=[rows[key] for key in ("truth","baseline","raw","initial")]
        em=np.load(staged/"element-matrix.npy")[:,active]
        for arm in plan["arms"]:
            record=next(r for r in result["records"] if r["arm"]==arm);weights=arrays(root/str(seed)/(arm+".npz"))
            # CPU FP64 recount of the FP32 affine gate; threshold decisions below
            # use its saved probabilities after this arithmetic agreement check.
            cx=(rows["coordinate"]/scale).astype(np.float32).astype(float)
            if arm=="nonlinear":
                values=[]
                for start in range(0,len(cx),1024):
                    x=rows["features"][start:start+1024].astype(float);q=cx[start:start+1024]
                    h=np.einsum("bf,sfh->bsh",x,weights["hidden_weight"].astype(float),optimize=True)+weights["hidden_bias"]
                    h+=q[:,:,None]*weights["hidden_own"][:,0]+abs(q)[:,:,None]*weights["hidden_own"][:,1]
                    h=.5*h*(1+erf(h/np.sqrt(2)))
                    values.append((h*weights["output_weight"]).sum(-1)+weights["bias"])
                logits=np.concatenate(values)
            else:logits=rows["features"].astype(float)@weights["weight"].astype(float)+weights["bias"]+cx*weights["own"][0]+abs(cx)*weights["own"][1]
            prob=rows[arm+"_probability"];recomputed=np.exp(-np.logaddexp(0,-logits))
            np.testing.assert_allclose(prob,recomputed,rtol=1e-4,atol=1e-6)
            threshold=choose_independently(prob[20000:30000],b[20000:30000],c[20000:30000],initial[20000:30000],d[20000:30000])
            require(abs(threshold-record["threshold"])<1e-14,"Calibration threshold differs")
            chosen=(prob>=record["threshold"])&(initial+c>=0)
            np.testing.assert_array_equal(chosen,rows[arm+"_selected"]);np.testing.assert_array_equal(np.where(chosen,c,b),rows[arm])
            for name,sl in dict(fitting=slice(0,20000),calibration=slice(20000,30000),check=slice(30000,50000)).items():
                y=rows[arm][sl];truth=d[sl];base=b[sl];tau=1e-15+.1*abs(truth);old=abs(base-truth)<=tau;new=abs(y-truth)<=tau;nt=abs(truth)>1e-15/.9
                fullprediction=original["base"][sl].copy();fullprediction[:,active]=y
                population={key:value[sl] for key,value in original.items()}
                scored=score(dict(prediction=fullprediction,source_indices=population["source_indices"]),population,binding["species"],f"{seed}:{arm}:{name}")
                rawfull=population["base"].copy();rawfull[:,active]=c[sl]
                rawscore=score(dict(prediction=rawfull,source_indices=population["source_indices"]),population,binding["species"],f"{seed}:raw:{name}")
                base_score=score(dict(prediction=population["base"],source_indices=population["source_indices"]),population,binding["species"],f"{seed}:base:{name}")
                expected=record["populations"][name];g=scored["groups"]["all"]
                repaired=int((new&~old).sum());damaged=int((~new&old).sum())
                require(repaired==g["repaired"] and damaged==g["damaged"] and int(new.sum())==g["pass_count"]==expected["groups"]["all"]["passes"],"Pass counts differ")
                R=repaired/(~old).sum();D=damaged/old.sum()
                require(abs(new.mean()-(old.mean()+(1-old.mean())*R-old.mean()*D))<3e-16,"Repair identity differs")
                signs=int(((np.sign(y)!=np.sign(truth))&nt).sum());negative=int(((initial[sl]+y)<0).sum())
                require(signs==scored["nontrivial_sign"]["wrong_sign_count"]==expected["sign_errors"] and negative==scored["endpoint_diagnostics"]["negative_components"]==expected["negative"],"Sign/endpoint counts differ")
                np.testing.assert_array_equal(new.sum(0),expected["species_passes"])
                np.testing.assert_allclose(np.quantile(abs(y-truth)/tau,.99),g["normalized_error"]["p99"],rtol=1e-14)
                row=dict(seed=seed,arm=arm,population=name,acceptance=float(new.mean()),base=float(old.mean()),raw=float((abs(c[sl]-truth)<=tau).mean()),repaired=repaired,damaged=damaged,damage_rate=float(D),threshold=record["threshold"],selected=int(chosen[sl].sum()),nontrivial=float(new[nt].mean()),base_nontrivial=float(old[nt].mean()),sign_errors=signs,base_sign_errors=int(((np.sign(base)!=np.sign(truth))&nt).sum()),negative=negative,p99=g["normalized_error"]["p99"],base_p99=base_score["groups"]["all"]["normalized_error"]["p99"],weak_p99=scored["groups"]["weak_nontrivial"]["normalized_error"]["p99"],base_weak_p99=base_score["groups"]["weak_nontrivial"]["normalized_error"]["p99"],mass_residual_p99=float(np.quantile(abs(y.sum(1)),.99)),element_residual_p99=float(np.quantile(abs(y@em.T).max(1),.99)))
                row["conservation_comparators"]={label:dict(mass_p99=float(np.quantile(abs(values.sum(1)),.99)),element_p99=float(np.quantile(abs(values@em.T).max(1),.99))) for label,values in (("base",base),("raw",c[sl]))}
                table.append(row);scores.append(dict(seed=seed,arm=arm,population=name,candidate=scored,base=base_score,raw=rawscore))
        if plan.get("control_run"):
            control=read_json(checked_file(args.project/plan["control_run"]/"review.json",plan["control_review_sha256"]))
            a=next(t for t in table if t["seed"]==seed and t["arm"]=="nonlinear" and t["population"]=="check")
            c0=next(t for t in control["table"] if t["seed"]==seed and t["arm"]=="affine" and t["population"]=="check")
        else:a,c0=[next(t for t in table if t["seed"]==seed and t["arm"]==arm and t["population"]=="check") for arm in ("affine","intercept")]
        passes.append(a["acceptance"]>=max(a["base"],c0["acceptance"])+.005 and a["nontrivial"]>=a["base_nontrivial"] and a["damage_rate"]<=.02 and a["sign_errors"]<=a["base_sign_errors"] and a["negative"]==0 and a["p99"]<=1.05*a["base_p99"] and a["weak_p99"]<=1.05*a["base_weak_p99"])
    save(root/"protected-scores.json",dict(records=scores));verify_evaluator_lock(args.project)
    save(root/"review.json",dict(status="verified",table=table,report_sha256=digest(root/"report.json"),score_sha256=digest(root/"protected-scores.json"),advance=all(passes),candidate=False,confirmation_opened=False))
    print(root/"review.json")


if __name__=="__main__":main()
