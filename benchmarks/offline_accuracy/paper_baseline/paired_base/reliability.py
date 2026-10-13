"""Training-only cross-fit of a small input-disagreement reliability table."""
import argparse
import time
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import score
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.extract import source_revision


def groups(base,other):
    if base.shape!=other.shape or not np.isfinite(base).all() or not np.isfinite(other).all():raise ValueError("Invalid base pair")
    valid=(base!=0)&(other!=0)&(np.sign(base)==np.sign(other))&(abs(other)<abs(base))
    result=np.full(base.shape,-1,dtype=np.int8)
    logratio=np.log(abs(base[valid]))-np.log(abs(other[valid]))
    result[valid]=np.searchsorted(np.log([1.1,2.,10.]),logratio,side="right")
    return result


def fit(base,other,truth,minimum=128):
    category=groups(base,other);table=np.zeros((base.shape[1],4),dtype=bool);records=[]
    allowance=1e-15+.1*abs(truth);p0=abs(base-truth)<=allowance;p1=abs(other-truth)<=allowance
    nontrivial=abs(truth)>1e-15/.9
    for species in range(base.shape[1]):
        for bucket in range(4):
            mask=category[:,species]==bucket;a=p0[:,species][mask];b=p1[:,species][mask];nt=nontrivial[:,species][mask]
            repaired=int((~a&b).sum());damaged=int((a&~b).sum());basepass=int(a.sum())
            ntchange=int(b[nt].sum())-int(a[nt].sum())
            selected=len(a)>=minimum and basepass>0 and repaired>damaged and damaged/basepass<=.02 and ntchange>=0
            table[species,bucket]=selected
            records.append(dict(species=species,bucket=bucket,count=len(a),basepass=basepass,repairs=repaired,damage=damaged,nontrivial_change=ntchange,selected=bool(selected)))
    return table,records


def predict(base,other,table):
    category=groups(base,other)
    if table.shape!=(base.shape[1],4) or table.dtype!=bool:raise ValueError("Invalid fixed table")
    selected=(category>=0)&table[np.arange(base.shape[1])[None,:],np.maximum(category,0)]
    return np.where(selected,other,base)


def main():
    from ..hidden_features.prepare import save
    from ..kinetic_features.review import recount
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--execute",action="store_true");a=p.parse_args()
    project=a.project;plan=read_json(project/"research/experiments/EXP-20261013-pair-reliability.json");root=project/plan["run"];source=project/plan["source_run"]
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");policy=read_json(project/"research/campaign.json")["screening"]
    review=read_json(checked_file(source/"review.json",plan["source_review_sha256"]));report=read_json(checked_file(source/"report.json",review["report_sha256"]))
    for name,sha in report["artifacts"].items():checked_file(source/name,sha)
    if root.exists():raise ValueError("Existing result; do not overwrite")
    if source_revision()["dirty"]:raise ValueError("Commit before execution")
    if not a.execute:print("PRECHECKED");return
    started=time.monotonic();root.mkdir(parents=True);save(root/"frozen-experiment.json",plan);save(root/"source.json",source_revision())
    active=np.array([s!="AR" for s in binding["species"]]);summaries=[];scores=[];allgates=[];foldgates=[]
    permutation=np.random.default_rng(plan["split_seed"]).permutation(50000);folds=np.empty(50000,dtype=int);folds[permutation[:25000]]=0;folds[permutation[25000:]]=1
    common=None
    for seed in plan["seeds"]:
        rows=arrays(source/"prepared"/str(seed)/"training-reference.npz");inputs=arrays(source/"prepared"/str(seed)/"fit-input/training.npz")
        assert len(rows["states"])==50000
        for key,value in rows.items():np.testing.assert_array_equal(value,inputs[key])
        if common is None:common=rows["source_indices"]
        else:np.testing.assert_array_equal(common,rows["source_indices"])
        base=rows["base"];other=inputs["other"];pred=base.copy();tables=[];training_records=[]
        full_other=np.where(groups(base,other)>=0,other,base)
        for fold in range(2):
            train=folds!=fold;check=folds==fold
            table,details=fit(base[train],other[train],rows["delta"][train]);tables.append(table);training_records.append(details)
            pred[check]=predict(base[check],other[check],table)
            for role,mask in (("fitting",train),("check",check)):
                subset={k:v[mask] for k,v in rows.items()};y=predict(base[mask],other[mask],table)
                result=score(dict(prediction=y,source_indices=subset["source_indices"]),subset,binding["species"],f"{seed}:fold{fold}:{role}")
                recount(y,subset["delta"],subset["base"],result,active,subset["states"][:,2:]);g=result["groups"]["all"]
                summaries.append(dict(seed=seed,fold=fold,population=role,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],selected_bins=int(table[active].sum())))
                if role=="check":foldgates.append(g["acceptance"]>g["base_acceptance"] and g["damage_rate_among_base_successes"]<=.02)
                scores.append(dict(seed=seed,fold=fold,population=role,result=result))
        for arm,y in (("crossfit",pred),("unconditional",full_other)):
            result=score(dict(prediction=y,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{arm}")
            baseline=score(dict(prediction=base,source_indices=rows["source_indices"]),rows,binding["species"],f"{seed}:{arm}")
            recount(y,rows["delta"],base,result,active,rows["states"][:,2:]);gate=comparison(result,baseline,policy);g=result["groups"]["all"]
            summaries.append(dict(seed=seed,population=arm,acceptance=g["acceptance"],base=g["base_acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],gate=gate))
            if arm=="crossfit":allgates.append(gate["passed"])
            scores.append(dict(seed=seed,population=arm,result=result,baseline=baseline))
        np.savez(root/f"{seed}.npz",source_indices=rows["source_indices"],folds=folds,prediction=pred,unconditional=full_other,tables=np.stack(tables))
        save(root/f"{seed}-fit.json",dict(records=training_records))
    verify_evaluator_lock(project)
    if time.monotonic()-started>plan["resources"]["maximum_seconds"]:raise TimeoutError("Diagnostic budget")
    save(root/"scores.json",dict(records=scores))
    report=dict(status="verified",summary=summaries,proceed_full_screen=bool(all(allgates) and all(foldgates)),elapsed_seconds=time.monotonic()-started,
        limits="Training-only crossfit; pooled predictions are two different fitted tables, not one deployed model. No confirmation or promotion.",
        artifacts={p.name:digest(p) for p in root.iterdir() if p.is_file()})
    save(root/"review.json",report);print(summaries);print("proceed_full_screen",report["proceed_full_screen"])


if __name__=="__main__":main()
