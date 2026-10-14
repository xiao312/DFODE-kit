"""Independent CPU forward samples and full protected physical recount."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.special import erf
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import inputs,score
from benchmarks.research_contract.gates import comparison
from ..hidden_features.prepare import save
from ..kinetic_features.review import recount,require


def independent_coordinate(weights,x,arm):
    w={k:v.cpu().double().numpy() for k,v in weights.items()}
    gelu=lambda v:.5*v*(1+erf(v/np.sqrt(2)))
    if arm=="shared":
        return gelu(gelu(x@w['0.weight'].T+w['0.bias'])@w['2.weight'].T+w['2.bias'])@w['4.weight'].T+w['4.bias']
    result=[]
    for species in range(58):
        h=gelu(x@w['w1'][species]+w['b1'][species])
        h=gelu(h@w['w2'][species]+w['b2'][species])
        result.append(h@w['w3'][species]+w['b3'][species])
    return np.stack(result,axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--run",type=Path,required=True);args=p.parse_args()
    root=args.run;project=args.project
    if (root/"review.json").exists():raise ValueError("Existing review")
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");inputs(project,binding)
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,sha in report["artifacts"].items():checked_file(root/path,sha)
    plan=read_json(root/"frozen-experiment.json");policy=read_json(project/"research/campaign.json")["screening"]
    active=np.array([s!="AR" for s in binding["species"]]);table=[];scores=[];advance=[]
    for seed in plan["seeds"]:
        staged=root/"prepared"/str(seed);original=arrays(staged/"training.npz");saved=arrays(root/str(seed)/"inference/predictions.npz")
        full=arrays(project/binding["prepared_root"]/str(seed)/"training.npz")
        unseen=arrays(project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
        allowed=np.setdiff1d(np.arange(len(full["states"])),np.r_[np.arange(10000),unseen]);selected=np.random.default_rng(seed+701).permutation(allowed)[:50000]
        np.testing.assert_array_equal(selected,np.load(staged/"selected-positions.npy"))
        for key,value in original.items():np.testing.assert_array_equal(value,full[key][selected])
        del full
        for key,value in arrays(staged/"fit-input/training.npz").items():np.testing.assert_array_equal(value,original[key][:20000])
        stateonly=arrays(staged/"inference-input/states.npz");require(set(stateonly)=={"states","source_indices"},"Inference boundary")
        for key,value in stateonly.items():np.testing.assert_array_equal(value,original[key])
        np.testing.assert_array_equal(saved["source_indices"],original["source_indices"]);np.testing.assert_array_equal(saved["base"],original["base"])
        records=read_json(root/str(seed)/"result.json")["records"];timings=read_json(root/str(seed)/"inference/result.json")["records"]
        for arm in plan["arms"]:
            directory=root/str(seed)/arm;fit_record=next(r for r in records if r["arm"]==arm)
            require(fit_record["updates_completed"]==6000 and fit_record["row_presentations"]==60000000 and fit_record["parameter_count"]==plan["parameters"][arm],"Actual work differs")
            replay=arrays(directory/"training-replay.npz");np.testing.assert_array_equal(replay["prediction"],saved[arm][:20000]);np.testing.assert_array_equal(replay["coordinate"],saved[arm+"_coordinate"][:20000])
            checked_file(staged/"inference-model"/arm/"weights.pt",digest(directory/"weights.pt"))
            weights=torch.load(directory/"weights.pt",weights_only=True,map_location="cpu")
            ix=np.linspace(0,49999,32,dtype=int);q=independent_coordinate(weights,original["features"][ix].astype(float),arm)
            np.testing.assert_allclose(q,saved[arm+"_coordinate"][ix],rtol=1e-4,atol=2e-6)
            oracle=1e-14*np.sinh(np.arcsinh(saved["base"][:,active]/1e-14)+saved[arm+"_coordinate"])
            require(float(np.max(abs(oracle-saved[arm][:,active])/(1e-15+.1*abs(original["delta"][:,active]))))<1e-5,"Decoder arithmetic mismatch")
            for population,sl in dict(fitting=slice(0,20000),calibration=slice(20000,30000),check=slice(30000,None)).items():
                rows={k:v[sl] for k,v in original.items()};pred=saved[arm][sl]
                candidate=score(dict(prediction=pred,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{arm}:{population}")
                baseline=score(dict(prediction=rows["base"],source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:base:{population}")
                recount(pred,rows["delta"],rows["base"],candidate,active,rows["states"][:,2:]);g=candidate["groups"]["all"]
                row=dict(seed=seed,arm=arm,population=population,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],p99=g["normalized_error"]["p99"],negative=candidate["endpoint_diagnostics"]["negative_components"],gate=comparison(candidate,baseline,policy),latency_ratio=next(t["ratio"] for t in timings if t["arm"]==arm))
                table.append(row);scores.append(dict(seed=seed,arm=arm,population=population,candidate=candidate,base=baseline))
        check={t["arm"]:t for t in table if t["seed"]==seed and t["population"]=="check"}
        advance.append(check["species"]["gate"]["passed"] and check["species"]["acceptance"]>=check["shared"]["acceptance"]+.005)
    save(root/"protected-scores.json",dict(records=scores));verify_evaluator_lock(project)
    save(root/"review.json",dict(status="verified",advance=all(advance),table=table,report_sha256=digest(root/"report.json"),score_sha256=digest(root/"protected-scores.json"),candidate=False,confirmation_opened=False,qualification="Training-pool diagnostic only. Latency failure is accuracy-only research evidence, not promotion."))
    print(root/"review.json")


if __name__=="__main__":main()
