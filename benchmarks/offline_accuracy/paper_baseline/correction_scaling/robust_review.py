"""Recount the new coordinate Smooth L1 arm against unchanged historical controls."""
import argparse
import json
from pathlib import Path
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from .run import prepared_manifest
from .review import recount,progression
from .plan import SEEDS,configuration


def paired(new,old):
    expected = dict(old["config"],fit_objective="coordinate-sl1",
                    coordinate_beta=configuration(old["config"]["seed"],10000,"coordinate-sl1")["coordinate_beta"])
    if new["config"]!=expected:
        raise ValueError("New arm changed settings beyond its objective")
    for key in ("initial_weights_sha256","prepared_manifest_sha256","continued_result_sha256",
                "training_count","development_count","row_presentations","updates_completed","parameter_count"):
        if new[key]!=old[key]:
            raise ValueError("Unmatched historical control: "+key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Review output must be new")
    report = dict(status="verified",source=source_revision(),size=10000,fits=[],progression=[])
    for seed in SEEDS:
        manifest = prepared_manifest(args.original/"prepared",seed)
        prepared = args.original/"prepared"/str(seed)
        summaries,results = {},{}
        environments = []
        for objective in ("physical","direct","coordinate-sl1"):
            parent = args.candidate if objective=="coordinate-sl1" else args.original/"10000"
            path = parent/f"{seed}-{objective}"
            summary,result = recount(path,prepared,10000,manifest["species"])
            environments.append(json.loads((path/"environment.json").read_text()))
            if result["config"]!=configuration(seed,10000,objective):
                raise ValueError("Fit configuration changed")
            summaries[objective],results[objective] = summary,result
            summary.update(raw_gradient_norm=result["raw_gradient_norm"],
                           gradient_clipping_count=result["gradient_clipping_count"],
                           source=result["source"])
            group_scores = json.loads((path/"group-scores.json").read_text())
            summary["conditional"] = {split:dict(
                sign=result[split]["sign"],overprediction=result[split]["overprediction"],
                magnitude_bins=group_scores[split]["magnitude_bins"],
                groups=group_scores[split]["groups"]) for split in ("training","development")}
            report["fits"].append(summary)
        if any(environment!=environments[0] for environment in environments[1:]):
            raise ValueError("Recorded runtime settings differ from historical controls")
        paired(results["coordinate-sl1"],results["direct"])
        for key in ("initial_weights_sha256","prepared_manifest_sha256","row_presentations"):
            if results["physical"][key]!=results["direct"][key]:
                raise ValueError("Historical objectives are not paired")
        report["progression"].append(dict(seed=seed,passed=progression(summaries["physical"],summaries["coordinate-sl1"])))
    report["numerical_gate_passed"] = all(r["passed"] for r in report["progression"])
    report["identical_recorded_environments"] = True
    report["advance"] = False
    report["decision_scope"] = "10k diagnostic only; preservation and conditional tails require review before a new size protocol"
    save(args.output,report)
    print(json.dumps(dict(status="verified",numerical_gate_passed=report["numerical_gate_passed"],
        fits=[dict(seed=r["seed"],objective=r["objective"],training=r["training"]["acceptance"],
                   development=r["development"]["acceptance"],damage=r["development"]["pass_to_fail"],
                   repairs=r["development"]["fail_to_pass"]) for r in report["fits"]])))


if __name__=="__main__":
    main()
