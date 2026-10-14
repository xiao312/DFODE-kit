"""Trusted label alignment, separate arithmetic recount and protected scoring."""
import argparse
from pathlib import Path
import numpy as np
import cantera as ct
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, verify_evaluator_lock
from benchmarks.research_contract.replay import score, inputs
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.chemistry import set_state
from ..hidden_features.prepare import save


def require(condition, message):
    if not condition: raise ValueError(message)


def recount(pred, truth, base, scored, active, initial):
    """Do not use worker helpers or scorer internals for result-defining counts."""
    y,d,b=pred[:,active],truth[:,active],base[:,active]
    tolerance=1e-15+.1*np.abs(d); old=np.abs(b-d)<=tolerance; passed=np.abs(y-d)<=tolerance
    repairs=int((~old & passed).sum()); damage=int((old & ~passed).sum()); g=scored["groups"]["all"]
    require(int(passed.sum())==g["pass_count"] and repairs==g["repaired"] and damage==g["damaged"],"Independent pass counts differ")
    R=repairs/int((~old).sum()); D=damage/int(old.sum())
    require(abs(passed.mean()-(old.mean()+(1-old.mean())*R-old.mean()*D))<3e-16,"Repair identity differs")
    nt=np.abs(d)>1e-15/.9
    masks=dict(all=np.ones_like(nt),nontrivial=nt,floor_dominated=~nt,
               weak_nontrivial=nt&(abs(d)<=1e-12),active_increment=abs(d)>1e-12)
    for name,mask in masks.items():
        group=scored["groups"][name]
        require(int(mask.sum())==group["components"] and int(passed[mask].sum())==group["pass_count"],"Group counts differ")
        require(int((~old&passed&mask).sum())==group["repaired"] and int((old&~passed&mask).sum())==group["damaged"],"Group transitions differ")
        if mask.any():np.testing.assert_allclose(np.quantile((abs(y-d)/tolerance)[mask],.99),group["normalized_error"]["p99"],rtol=1e-13)
    require(int(((np.sign(y)!=np.sign(d))&nt).sum())==scored["nontrivial_sign"]["wrong_sign_count"],"Sign counts differ")
    require(int(((initial+pred)<0).sum())==scored["endpoint_diagnostics"]["negative_components"],"Negative count differs")
    for event in scored["small_target_events"]:
        subset=abs(d)<=event["epsilon"]
        require(int(subset.sum())==event["components"] and int((subset&(abs(y)>=event["eta"])).sum())==event["events"],"Event counts differ")
    np.testing.assert_allclose(np.quantile(abs(y-d)/tolerance,.99),g["normalized_error"]["p99"],rtol=1e-13)
    for j,name in enumerate(scored["per_species"]):
        require(int(passed[:,j].sum())==scored["per_species"][name]["pass_count"],"Species counts differ")


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--project",type=Path,required=True); parser.add_argument("--run",type=Path,required=True); args=parser.parse_args()
    root=args.run; project=args.project
    if (root/"review.json").exists():parser.error("Review exists; do not overwrite")
    verify_evaluator_lock(project); binding=read_json(project/"research/bindings.json"); inputs(project,binding)
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,expected in report["artifacts"].items():checked_file(root/path,expected)
    plan=read_json(root/"frozen-experiment.json"); policy=read_json(project/"research/campaign.json")["screening"]
    active=np.array([s!="AR" for s in binding["species"]]); table=[]; scores=[]; advance=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed); original=arrays(staged/"training.npz")
        full=arrays(project/binding["prepared_root"]/str(seed)/"training.npz")
        unseen=arrays(project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
        allowed=np.setdiff1d(np.arange(len(full["states"])),np.r_[np.arange(10000),unseen])
        positions=np.random.default_rng(seed+701).permutation(allowed)[:50000]
        np.testing.assert_array_equal(positions,np.load(staged/"selected-positions.npy"))
        for key,value in original.items():np.testing.assert_array_equal(value,full[key][positions])
        del full
        for key,value in arrays(staged/"fit-input/training.npz").items():np.testing.assert_array_equal(value,original[key][:20000])
        only_states=arrays(staged/"inference-input/states.npz"); require(set(only_states)=={"states","source_indices"},"Inference label boundary")
        for key,value in only_states.items():np.testing.assert_array_equal(value,original[key])
        rows=arrays(root/str(seed)/"inference/predictions.npz"); np.testing.assert_array_equal(rows["source_indices"],original["source_indices"])
        np.testing.assert_array_equal(rows["base"],original["base"])
        rates=arrays(root/str(seed)/"rates.npz"); scaler=arrays(root/str(seed)/"scaler.npz")
        np.testing.assert_array_equal(rows["rate_coordinate"][:20000],rates["coordinate"])
        np.testing.assert_array_equal(rows["rate_raw"][:20000],rates["raw"])
        np.testing.assert_array_equal(scaler["mean"],rates["coordinate"].mean(0))
        np.testing.assert_array_equal(scaler["scale"],np.maximum(rates["coordinate"].std(0),1e-6))
        # Independent formula audit on 32 prespecified evenly spaced states.
        mechanism=checked_file(staged/"model/mechanism.yaml",binding["mechanism_sha256"])
        gas=ct.Solution(str(mechanism)); require(gas.species_names==binding["species"],"Species axis")
        for index in np.linspace(0,49999,32,dtype=int):
            x=original["states"][index]; set_state(gas,dict(T=x[0],P=x[1],Y=x[2:]))
            net=gas.net_production_rates*gas.molecular_weights/gas.density_mass
            loss=gas.destruction_rates*gas.molecular_weights/gas.density_mass
            z=np.divide(1e-6*loss,x[2:],out=np.zeros_like(loss),where=x[2:]>0)
            factor=np.ones_like(z); nonzero=z!=0; factor[nonzero]=-np.expm1(-z[nonzero])/z[nonzero]
            expected=np.r_[(1e-6*net)[active],(1e-6*net*factor)[active]]
            np.testing.assert_allclose(rows["rate_raw"][index],expected,rtol=5e-13,atol=1e-30)
        # Score raw feature approximations on these same rows. They are controls,
        # not newly trained or protected-U/D-qualified predictors.
        raw_scores=[]
        for mode,columns in (("euler",slice(0,58)),("frozen-exponential",slice(58,116))):
            estimate=np.zeros_like(original["base"]); estimate[:,active]=rows["rate_raw"][:,columns]
            for population,sl in dict(fitting=slice(0,20000),check=slice(30000,None)).items():
                subset={k:v[sl] for k,v in original.items()}
                rawscore=score(dict(prediction=estimate[sl],source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:{mode}:{population}")
                recount(estimate[sl],subset["delta"],subset["base"],rawscore,active,subset["states"][:,2:])
                raw_scores.append(dict(seed=seed,arm=mode,population=population,candidate=rawscore))
        scores.extend(raw_scores)
        timings=read_json(root/str(seed)/"inference/result.json")["records"]
        for arm in plan["arms"]:
            trained=root/str(seed)/arm; replay=arrays(trained/"training-replay.npz")
            actual=next(r for r in read_json(root/str(seed)/"result.json")["records"] if r["arm"]==arm)
            require(actual["updates_completed"]==6000 and actual["row_presentations"]==60000000 and actual["parameter_count"]==plan["parameters"][arm],"Actual fitting work differs")
            np.testing.assert_array_equal(replay["prediction"],rows[arm][:20000]); np.testing.assert_array_equal(replay["coordinate"],rows[arm+"_coordinate"][:20000])
            checked_file(staged/"inference-model"/arm/"weights.pt",digest(trained/"weights.pt"))
            config=read_json(trained/"config.json")
            require(config["updates"]==6000 and config["batch_size"]==10000 and config["training_count"]==20000,"Work changed")
            # Independent exact-arithmetic formula; stable decoder remains authority
            # near cancellation, so compare in physical allowance units.
            oracle=1e-14*np.sinh(np.arcsinh(rows["base"][:,active]/1e-14)+rows[arm+"_coordinate"])
            require(float(np.max(abs(oracle-rows[arm][:,active])/(1e-15+.1*abs(original["delta"][:,active]))))<1e-5,"Decoder mismatch")
            for population,sl in dict(fitting=slice(0,20000),calibration=slice(20000,30000),check=slice(30000,None)).items():
                subset={k:v[sl] for k,v in original.items()}; pred=rows[arm][sl]
                candidate=score(dict(prediction=pred,source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:{arm}:{population}")
                baseline=score(dict(prediction=subset["base"],source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:base:{population}")
                recount(pred,subset["delta"],subset["base"],candidate,active,subset["states"][:,2:])
                gate=comparison(candidate,baseline,policy); g=candidate["groups"]["all"]
                row=dict(seed=seed,arm=arm,population=population,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],nontrivial=candidate["groups"]["nontrivial"]["acceptance"],p99=g["normalized_error"]["p99"],negative=candidate["endpoint_diagnostics"]["negative_components"],gate=gate,latency_ratio=next(t["ratio"] for t in timings if t["arm"]==arm))
                table.append(row); scores.append(dict(seed=seed,arm=arm,population=population,candidate=candidate,base=baseline))
        checks={t["arm"]:t for t in table if t["seed"]==seed and t["population"]=="check"}
        advance.append(checks["kinetic"]["gate"]["passed"] and all(checks["kinetic"]["acceptance"]>checks[c]["acceptance"] for c in ("ordinary","capacity")))
    save(root/"protected-scores.json",dict(records=scores)); verify_evaluator_lock(project)
    kinetic_timing=all(t["latency_ratio"]<=policy["maximum_inference_latency_ratio"] for t in table if t["arm"]=="kinetic")
    save(root/"review.json",dict(status="verified",table=table,advance=all(advance),latency_within_anchor_budget=kinetic_timing,
        qualification="Training-pool diagnostic only; accuracy-only if latency exceeds budget; U/D not evaluated",candidate=False,confirmation_opened=False,report_sha256=digest(root/"report.json"),score_sha256=digest(root/"protected-scores.json")))
    print(root/"review.json")


if __name__=="__main__":main()
