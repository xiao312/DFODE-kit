"""Export aggregate evidence and a private source receipt after independent review."""
import argparse
from pathlib import Path
from benchmarks.research_contract.adapter import read_json,checked_file,digest
from ..local_predictability.summary import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--receipt",type=Path)
    args = p.parse_args()
    if args.output.exists() or (args.receipt and args.receipt.exists()):
        p.error("Use new output files")
    review = read_json(args.run/"review.json")
    report = read_json(checked_file(args.run/"report.json",review["report_sha256"]))
    if review["status"] != "verified":
        raise ValueError("Independent review required")
    records = []
    for row in report["records"]:
        full = row["score"]
        score = {k:full[k] for k in ("groups","coordinate_error","by_snapshot","nontrivial_sign",
                 "endpoint_diagnostics","returned_prediction_finite","all_active_species_state_acceptance")}
        score["per_species"] = {name:{k:g[k] for k in ("components","pass_count","base_pass_count","repaired","damaged")}
                                for name,g in full["per_species"].items()}
        score["small_target_events"] = [{k:event[k] for k in ("epsilon","eta","components","events","baseline_events")}
                                        for event in full["small_target_events"]]
        records.append({k:row[k] for k in ("seed","space","library","k","mode","population")}
                       | {"score":score})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,dict(experiment="EXP-20261013-species-distance",records=records,
         checks=report["checks"],timings=report["timings"],elapsed_seconds=report["elapsed_seconds"],
         report_sha256=digest(args.run/"report.json"),review_sha256=digest(args.run/"review.json"),
         neural_fits=0,candidate_promoted=False))
    if args.receipt:
        selected = []
        for row in records:
            if row["library"] != 50000 or row["mode"] != "correction" or row["population"] == "training_loo":
                continue
            g = row["score"]["groups"]["all"]
            selected.append({k:row[k] for k in ("seed","space","k","population")}
                            | {k:g[k] for k in ("components","pass_count","base_pass_count","repaired","damaged")})
        save(args.receipt,dict(schemaVersion=1,items=[dict(id="species-distance",title="Does species-specific distance improve correction transfer?",queries=[dict(
            id="metric-comparison",source=dict(label="DFODE species-distance diagnostic",
                files=[dict(label=args.output.name)],
                links=[dict(label="Research results",url="https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/research/RESULTS.md")],
                metricDefinitions=[dict(label="Physical acceptance",definition="A signed increment passes when absolute error is at most 1e-15 plus 0.1 times reference magnitude.")],
                filters=["Library: 50,000 fixed training states", "Queries: 2,048 same-source unseen and 1,023 development states per seed", "Species: 58 active outputs"],
                caveats=["Metric weights use only training-library labels, excluding the training query rows.","Diagnostic transfers are not promoted predictors. These query populations are not independent confirmation."],
                evidenceFlow=[dict(kind="validation",title="Independent recount",detail=f"Recounted all {len(review['records'])} physical pass, base-pass, repair and damage records. Recomputed global graphs exactly match the previous diagnostic.")]),
            columns=["seed","space","k","population","components","pass_count","base_pass_count","repaired","damaged"],rows=selected,
            preview=dict(kind="aggregate",note="Fixed 50k correction-transfer slice; all other registered comparisons remain in the linked evidence."),
            methods=[dict(language="calculation",code="acceptance = pass_count / components\nnet pass change = repaired - damaged")])])]))
    print(args.output)


if __name__ == "__main__":
    main()
