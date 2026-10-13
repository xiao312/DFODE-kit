"""Independent saved-evidence checks and compact handoff; no new scientific fits."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from ..correction_scaling.run import prepared_manifest,load_arrays
from ..decoder_loss.coordinates import decode
from .prepare import checked_plan


def checked_report(path):
    report = json.loads((path/"report.json").read_text())
    if report["status"]!="complete" or json.loads((path/"verification.json").read_text())["report_sha256"]!=sha256(path/"report.json"):
        raise ValueError("Diagnostic incomplete or changed")
    for name,digest in report["artifacts"].items():
        if Path(name).name!=name or sha256(path/name)!=digest:
            raise ValueError("Diagnostic artifact changed")
    return report


def reference_review(root,plan):
    report = checked_report(root/"audit")
    selection = load_arrays(root/"plan/selections.npz")
    active = np.array([s!="AR" for s in plan["species"]])
    for record in report["records"]:
        i = record["index"]
        raw = json.loads((root/"audit"/f"record-{i:03d}.json").read_text())
        ref = np.asarray(raw["checks"]["direct-tighter"]["delta"])
        initial,ordinary = selection["audit_states"][i,2:],selection["audit_ordinary"][i]
        deltas = np.asarray([v["delta"] for v in raw["checks"].values()])
        discrepancy = abs(ordinary-ref)/(1e-15+.1*abs(ref))
        uncertainty = np.maximum.reduce([abs(deltas-ref).max(axis=0),
            abs(np.spacing(initial)),abs(np.spacing(initial+ref))])
        np.testing.assert_array_equal(discrepancy,record["scores"]["nominal_difference_in_allowances"])
        np.testing.assert_array_equal(uncertainty,record["scores"]["uncertainty"])
        np.testing.assert_array_equal(uncertainty<=.1*(1e-15+.1*abs(ref)),record["scores"]["qualified"])
    summary = {}
    for role in ("tail","control"):
        rows = [r for r in report["records"] if r["selection"]["role"]==role]
        anchor = [r["scores"]["anchor"] for r in rows]
        summary[role] = dict(states=len(rows),qualified_anchors=sum(r["qualified"] for r in anchor),
            stable_anchor_signs=sum(r["sign_stable"] for r in anchor),
            nominal_anchor_sign_matches=sum(r["nominal_sign_match"] for r in anchor),
            base_anchor_sign_reversals=sum(r["base_sign_reversal"] for r in anchor),
            max_anchor_difference=max(r["nominal_difference_in_allowances"] for r in anchor),
            max_anchor_uncertainty=max(r["uncertainty_in_allowances"] for r in anchor),
            active_components=len(rows)*int(active.sum()),qualified_active=sum(r["scores"]["qualified_active"] for r in rows),
            nominal_over_tenth=sum(r["scores"]["nominal_over_tenth"] for r in rows),
            max_all_difference=max(r["scores"]["max_nominal_difference"] for r in rows))
    return dict(elapsed_seconds=report["elapsed_seconds"],groups=summary)


def transfer_review(root,original,plan):
    report = checked_report(root/"evaluation")
    selection = load_arrays(root/"plan/selections.npz")
    active = np.array([s!="AR" for s in plan["species"]])
    output = []
    for seed_report in report["seeds"]:
        seed = seed_report["seed"]
        prepared_manifest(original/"prepared",seed)
        full = load_arrays(original/"prepared"/str(seed)/"training.npz")
        for name,population in seed_report["populations"].items():
            if name=="development":
                rows = load_arrays(original/"prepared"/str(seed)/"development.npz")
            else:
                ix = slice(0,10000) if name=="training" else selection[f"unseen_{seed}"]
                rows = {k:v[ix] for k,v in full.items()}
            saved = load_arrays(root/"evaluation"/f"{seed}-{name}.npz")
            np.testing.assert_array_equal(saved["source_indices"],rows["source_indices"])
            b,d,c = rows["base"][:,active],rows["delta"][:,active],saved["coordinate"]
            allowance = 1e-15+.1*abs(d)
            old = abs(b-d)<=allowance
            curves = []
            for curve in population["curves"]:
                p = decode(b,curve["gamma"]*c,dict(decoder="asinh",scale_floor=1e-14))
                passed = abs(p-d)<=allowance
                damage,repair = int((old&~passed).sum()),int((~old&passed).sum())
                if damage!=curve["pass_to_fail"] or repair!=curve["fail_to_pass"] or float(passed.mean())!=curve["acceptance"]:
                    raise ValueError("Damping counts do not reproduce")
                if curve["gamma"]==1:
                    np.testing.assert_array_equal(p,saved["prediction"][:,active])
                curves.append({k:v for k,v in curve.items() if k!="groups"})
            bins = population["preservation"]["bins"]
            if sum(r["damaged"] for r in bins)!=curves[-1]["pass_to_fail"]:
                raise ValueError("Damage bins do not partition base successes")
            output.append(dict(seed=seed,population=name,states=len(rows["states"]),curves=curves,
                feature_calibration_seen_states=population.get("feature_calibration_seen_states"),
                preservation=population["preservation"],
                damaged_away_from_center=sum(r["away_from_reference_center"] for r in bins),
                damaged_overshoot=sum(r["same_direction_overshoot"] for r in bins)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","original","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Review output must be new")
    plan = checked_plan(args.root/"plan")
    result = dict(status="verified",source=source_revision(),plan_sha256=sha256(args.root/"plan/plan.json"),
        audit=reference_review(args.root,plan),populations=transfer_review(args.root,args.original,plan),
        report_hashes={name:sha256(args.root/name/"report.json") for name in ("audit","evaluation")})
    save(args.output,result)
    print(json.dumps(dict(status="verified",audit=result["audit"],populations=len(result["populations"]))))


if __name__=="__main__":
    main()
