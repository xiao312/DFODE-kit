"""Independent saved-count review and initial loss concentration; no new training."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from ..matched_work.run import checked_result
from .plan import SEEDS,SIZES,OBJECTIVES
from .run import prepared_manifest,load_arrays


def fitting_pressure(target,base_pass):
    square = target**2
    flat = square.ravel()
    count,total = max(1,len(flat)//100),float(flat.sum())
    return dict(components=len(flat),top_component_count=count,
        top_squared_loss_share=float(np.partition(flat,-count)[-count:].sum()/total) if total else None,
        base_passing_squared_loss_share=float(square[base_pass].sum()/total) if total else None,
        note="Initial squared-loss mass, not parameter-gradient mass or independent state count")


def recount(path,prepared,size,species):
    result = checked_result(path)
    if result["prepared_manifest_sha256"]!=sha256(prepared/"manifest.json"):
        raise ValueError("Result and prepared inputs differ")
    active = np.array([s!="AR" for s in species])
    output = dict(seed=result["config"]["seed"],objective=result["config"]["fit_objective"],
        result_sha256=sha256(path/"result.json"),seconds=result["training_wall_seconds"],history=result["history"])
    for split in ("training","development"):
        data = load_arrays(prepared/f"{split}.npz")
        n = size if split=="training" else len(data["delta"])
        truth,base,target = data["delta"][:n,active],data["base"][:n,active],data["target"][:n]
        saved = load_arrays(path/f"{split}-predictions.npz")
        np.testing.assert_array_equal(saved["source_indices"],data["source_indices"][:n])
        predicted,c = saved["prediction"][:,active],saved["coordinate"]
        allowance = 1e-15+.1*abs(truth)
        error = abs(predicted-truth)/allowance
        passed,old = error<=1,abs(base-truth)<=allowance
        nontrivial = abs(truth)>1e-15/.9
        expected = result[split]
        if (expected["all"]["pass_count"]!=int(passed.sum()) or
                expected["nontrivial"]["pass_count"]!=int(passed[nontrivial].sum()) or
                expected["whole_states_passed"]!=int(passed.all(axis=1).sum()) or
                expected["pass_to_fail"]!=int((old&~passed).sum()) or
                expected["fail_to_pass"]!=int((~old&passed).sum())):
            raise ValueError("Independent physical counts disagree")
        output[split] = dict(components=int(truth.size),acceptance=float(passed.mean()),
            base_acceptance=float(old.mean()),nontrivial_acceptance=float(passed[nontrivial].mean()),
            base_nontrivial_acceptance=float(old[nontrivial].mean()),
            normalized_error_p99=float(np.quantile(error,.99)),
            coordinate_rmse=float(np.sqrt(np.mean((c-target)**2))),
            base_coordinate_rmse=float(np.sqrt(np.mean(target**2))),
            coordinate_sufficient_fraction=float((abs(c-target)<=np.log1p(.1)).mean()),
            pass_to_fail=int((old&~passed).sum()),fail_to_pass=int((~old&passed).sum()),
            negative_endpoint_components=int((data["states"][:n,2:][:,active]+predicted<0).sum()))
        if split=="training":
            output[split]["fitting_pressure"] = fitting_pressure(target,old)
            output[split]["unique_fp32_feature_rows"] = len(np.unique(data["features"][:n],axis=0))
    return output,result


def progression(physical,direct):
    p,d = physical["development"],direct["development"]
    return bool(d["acceptance"]>=max(p["acceptance"],d["base_acceptance"])+.005 and
        d["nontrivial_acceptance"]>=max(p["nontrivial_acceptance"],d["base_nontrivial_acceptance"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--size",type=int,choices=SIZES,default=10000)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Review output must be new")
    report = dict(status="verified",source=source_revision(),size=args.size,fits=[],progression=[])
    for seed in SEEDS:
        manifest = prepared_manifest(args.root/"prepared",seed)
        pair = []
        for objective in OBJECTIVES:
            path = args.root/str(args.size)/f"{seed}-{objective}"
            summary,result = recount(path,args.root/"prepared"/str(seed),args.size,manifest["species"])
            pair.append((summary,result))
            report["fits"].append(summary)
        for key in ("initial_weights_sha256","prepared_manifest_sha256","training_count","row_presentations","source"):
            if pair[0][1][key]!=pair[1][1][key]:
                raise ValueError("Unpaired comparison: "+key)
        report["progression"].append(dict(seed=seed,passed=progression(pair[0][0],pair[1][0])))
    report["advance"] = all(r["passed"] for r in report["progression"])
    if args.output:
        save(args.output,report)
    print(json.dumps(dict(status=report["status"],fits=len(report["fits"]),advance=report["advance"])))


if __name__=="__main__":
    main()
