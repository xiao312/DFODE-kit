"""Protected scoring and independent CPU projection/count reconstruction."""
import argparse
from pathlib import Path
import cantera as ct
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score
from benchmarks.research_contract.gates import comparison
from ..hidden_features.prepare import save
from ..kinetic_features.review import recount,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True);args=p.parse_args()
    project,root=args.project,args.run
    if (root/"review.json").exists():raise ValueError("Existing review")
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");inputs(project,binding)
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,sha in report["artifacts"].items():checked_file(root/path,sha)
    plan=read_json(root/"frozen-experiment.json");policy=read_json(project/"research/campaign.json")["screening"]
    active=np.array([s!="AR" for s in binding["species"]]);table=[];scores=[];advance=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed);rows=arrays(staged/"training.npz");saved=arrays(root/str(seed)/"predictions.npz")
        full=arrays(project/binding["prepared_root"]/str(seed)/"training.npz")
        unseen=arrays(project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
        allowed=np.setdiff1d(np.arange(len(full["states"])),np.r_[np.arange(10000),unseen]);ix=np.random.default_rng(seed+701).permutation(allowed)[:50000]
        np.testing.assert_array_equal(ix,np.load(staged/"selected-positions.npy"))
        for key,value in rows.items():np.testing.assert_array_equal(value,full[key][ix])
        del full
        stateonly=arrays(staged/"inference-input/states.npz");require(set(stateonly)=={"states","source_indices"},"Inference boundary")
        for key,value in stateonly.items():np.testing.assert_array_equal(value,rows[key])
        np.testing.assert_array_equal(saved["base"],rows["base"]);np.testing.assert_array_equal(saved["source_indices"],rows["source_indices"])
        gas=ct.Solution(str(checked_file(staged/"model/mechanism.yaml",binding["mechanism_sha256"])))
        require(gas.species_names==binding["species"],"Species axis")
        a=np.array([[gas.n_atoms(k,j)*gas.atomic_weights[j]/gas.molecular_weights[k] for k in np.flatnonzero(active)] for j in range(gas.n_elements)])
        a=a[np.any(a!=0,axis=1)];np.testing.assert_array_equal(a,np.load(staged/"projection-model/element-matrix.npy"))
        timing=read_json(root/str(seed)/"result.json")["records"]
        for arm in plan["arms"]:
            discrepancy=0.
            for index in np.linspace(0,49999,32,dtype=int):
                b=rows["base"][index,active];scale=np.ones_like(b) if arm=="uniform" else 1e-14+abs(b);scale/=scale.max()
                u,s,vh=np.linalg.svd(a*scale,full_matrices=False);fallback=s[-1]<=1e-12*s[0]
                require(bool(saved[arm+"_fallback"][index])==fallback,"Rank policy differs")
                expected=b if fallback else b-scale*(vh.T@((u.T@(a@b))/s))
                error=abs(expected-saved[arm][index,active]);discrepancy=max(discrepancy,float(error.max()))
                require(float(error.max())<=1e-10*max(float(abs(b).max()),1e-14),"Independent projection differs")
            np.testing.assert_array_equal(saved[arm][saved[arm+"_fallback"]],rows["base"][saved[arm+"_fallback"]])
            for population,sl in dict(fitting=slice(0,20000),calibration=slice(20000,30000),check=slice(30000,None)).items():
                subset={k:v[sl] for k,v in rows.items()};pred=saved[arm][sl]
                candidate=score(dict(prediction=pred,source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:{arm}:{population}")
                baseline=score(dict(prediction=subset["base"],source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:base:{population}")
                recount(pred,subset["delta"],subset["base"],candidate,active,subset["states"][:,2:]);g=candidate["groups"]["all"]
                entry=dict(seed=seed,arm=arm,population=population,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],p99=g["normalized_error"]["p99"],negative=candidate["endpoint_diagnostics"]["negative_components"],fallback_states=int(saved[arm+"_fallback"][sl].sum()),element_residual_p99=float(np.quantile(abs(pred[:,active]@a.T).max(1),.99)),base_element_residual_p99=float(np.quantile(abs(subset["base"][:,active]@a.T).max(1),.99)),cpu_sample_max_difference=discrepancy,gate=comparison(candidate,baseline,policy),latency_ratio=next(t["ratio"] for t in timing if t["arm"]==arm))
                table.append(entry);scores.append(dict(seed=seed,arm=arm,population=population,candidate=candidate,base=baseline))
        advance.append(next(t for t in table if t["seed"]==seed and t["arm"]=="amplitude" and t["population"]=="check")["gate"]["passed"])
    save(root/"protected-scores.json",dict(records=scores));verify_evaluator_lock(project)
    save(root/"review.json",dict(status="verified",table=table,advance=all(advance),candidate=False,confirmation_opened=False,report_sha256=digest(root/"report.json"),score_sha256=digest(root/"protected-scores.json")))
    print(root/"review.json")


if __name__=="__main__":main()
