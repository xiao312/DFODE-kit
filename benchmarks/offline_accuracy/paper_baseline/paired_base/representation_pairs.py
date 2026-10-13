"""Matched-work representation complementarity on pinned training rows only."""
import argparse
import time
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest,verify_evaluator_lock
from benchmarks.research_contract.replay import score
from benchmarks.research_contract.gates import comparison
from benchmarks.flame_conditioning.extract import source_revision
from ..hidden_features.prepare import save
from ..kinetic_features.review import recount
from ..matched_targets.checks import check_result,check_pair
from .metrics import align_indices,agreement,bounded_mean,strict_shrink


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);p.add_argument("--execute",action="store_true");a=p.parse_args();project=a.project
    plan=read_json(project/"research/experiments/EXP-20261013-representation-pairs.json");root=project/plan["run"];source=project/plan["source_run"];campaign=project/plan["campaign"]
    verify_evaluator_lock(project);binding=read_json(project/"research/bindings.json");policy=read_json(project/"research/campaign.json")["screening"]
    prior=read_json(checked_file(source/"review.json",plan["source_review_sha256"]));report=read_json(checked_file(source/"report.json",prior["report_sha256"]))
    for name,sha in report["artifacts"].items():checked_file(source/name,sha)
    matrix=read_json(checked_file(campaign/"verification.json",plan["campaign_verification_sha256"]));identities={r["name"]:r["result_sha256"] for r in matrix["results"]}
    rows={s:arrays(source/"prepared"/str(s)/"training-reference.npz") for s in plan["seeds"]};predictions={};models=[]
    common=rows[plan["seeds"][0]]["source_indices"]
    for s in plan["seeds"]:
        np.testing.assert_array_equal(common,rows[s]["source_indices"]);first=None;firstprep=None;firstids=None
        for target in plan["targets"]:
            name=f"{s}-{target}-increment";directory=campaign/name;r=read_json(checked_file(directory/"result.json",identities[name]));check_result(r)
            assert r["hashes"]["dataset_manifest"]==binding["dataset_manifest_sha256"]
            for name2,sha in r["artifacts"].items():checked_file(directory/name2,sha)
            ids=np.load(directory/"training-indices.npy");prep=arrays(directory/"preprocessing.npz")
            if first is not None:
                check_pair(first,r,same_target=False);np.testing.assert_array_equal(ids,firstids)
                for key in ("x_offset","x_scale","x_constant","active"):np.testing.assert_array_equal(prep[key],firstprep[key])
            else:first,firstprep,firstids=r,prep,ids
            values=arrays(directory/"training-predictions.npz");np.testing.assert_array_equal(values["source_indices"],ids)
            ix=align_indices(ids,common);predictions[(s,target)]=values["prediction"][ix]
            models.append(dict(seed=s,target=target,result_sha256=identities[name],weights_sha256=r["artifacts"]["weights.pt"]))
    if root.exists():raise ValueError("Existing output")
    if source_revision()["dirty"]:raise ValueError("Commit before execution")
    if not a.execute:print("PRECHECKED");return
    started=time.monotonic();root.mkdir(parents=True);save(root/"source.json",source_revision());save(root/"frozen-experiment.json",plan)
    active=np.array([s!="AR" for s in binding["species"]]);summaries=[];scores=[];overlaps=[];gates={};data={"source_indices":common}
    for s in plan["seeds"]:
        row=rows[s];b=predictions[(s,"gbct")];d=row["delta"];matched=dict(row,base=b)
        baseline=score(dict(prediction=row["base"],source_indices=common),row,binding["species"],str(s));matched_score=score(dict(prediction=b,source_indices=common),matched,binding["species"],str(s))
        partner=next(seed for seed in plan["seeds"] if seed!=s)
        choices={t:predictions[(s,t)] for t in plan["targets"] if t!="gbct"};choices["gbct-other-seed"]=predictions[(partner,"gbct")]
        for target,q in choices.items():
            masks=dict(all=np.ones_like(d[:,active],dtype=bool),floor=abs(d[:,active])<=1e-15/.9,weak=(abs(d[:,active])>1e-15/.9)&(abs(d[:,active])<=1e-12),active=abs(d[:,active])>1e-12)
            overlaps.append(dict(seed=s,target=target,groups={k:agreement(b[:,active],q[:,active],d[:,active],m) for k,m in masks.items()}))
            candidates={"member":q,"mean":.5*b+.5*q,"strict":strict_shrink(b,bounded_mean(b,q))}
            for rule,y in candidates.items():
                data[f"{s}-{target}-{rule}"]=y
                result=score(dict(prediction=y,source_indices=common),row,binding["species"],f"{s}:{target}:{rule}");recount(y,d,row["base"],result,active,row["states"][:,2:])
                matched_result=score(dict(prediction=y,source_indices=common),matched,binding["species"],f"{s}:{target}:{rule}:matched");recount(y,d,b,matched_result,active,row["states"][:,2:])
                gate=comparison(result,baseline,policy);g=result["groups"]["all"]
                summaries.append(dict(seed=s,target=target,rule=rule,acceptance=g["acceptance"],continued_base=g["base_acceptance"],matched_base=matched_score["groups"]["all"]["acceptance"],repairs=g["repaired"],damage=g["damaged"],damage_rate=g["damage_rate_among_base_successes"],negative=result["endpoint_diagnostics"]["negative_components"],gate=gate))
                scores.append(dict(seed=s,target=target,rule=rule,continued_comparison=result,matched_comparison=matched_result))
                gates[(s,target,rule)]=gate["passed"]
    decisions=[]
    for target in plan["targets"][1:]:
        for rule in ("mean","strict"):
            def cell(s,t):return next(r for r in summaries if r["seed"]==s and r["target"]==t and r["rule"]==rule)
            wins=all(cell(s,target)["acceptance"]>cell(s,"gbct-other-seed")["acceptance"] for s in plan["seeds"])
            decisions.append(dict(target=target,rule=rule,beats_seed_control_both=bool(wins),advance=bool(wins and all(gates[(s,target,rule)] for s in plan["seeds"])) ))
    if time.monotonic()-started>plan["resources"]["maximum_seconds"]:raise TimeoutError("Finite diagnostic budget")
    np.savez(root/"predictions.npz",**data);save(root/"scores.json",dict(records=scores));verify_evaluator_lock(project)
    save(root/"review.json",dict(status="verified_cached_diagnostic",models=models,summary=summaries,overlaps=overlaps,decisions=decisions,elapsed_seconds=time.monotonic()-started,
        limits="All models saw training rows; means are fixed, no new learned recipe, no inference timing or independent confirmation. Continued base has extra training, so use matched comparator for representation conclusions.",artifacts={f.name:digest(f) for f in root.iterdir() if f.is_file()}))
    print(summaries);print(decisions)


if __name__=="__main__":main()
