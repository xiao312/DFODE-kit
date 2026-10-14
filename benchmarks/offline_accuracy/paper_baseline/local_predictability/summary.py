"""Export small, source-bound review data; never publish private state arrays."""
import argparse
import json
from pathlib import Path
from benchmarks.research_contract.adapter import read_json, checked_file, digest


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.receipt and args.receipt.exists()):
        parser.error("Use new output files")
    review = read_json(args.run / "review.json")
    report = read_json(checked_file(args.run / "report.json",review["report_sha256"]))
    if review["status"] != "verified":
        raise ValueError("Independent recount required")
    records = []
    for row in report["records"]:
        records.append({key:row[key] for key in ("seed","space","library","k","mode","shuffled","population")}
                       | {"groups":row["score"]["groups"], "coordinate_error":row["score"]["coordinate_error"],
                          "by_snapshot":row["score"]["by_snapshot"]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save(args.output,dict(experiment="EXP-20261013-local-predictability", scope=report["scope"],
                         report_sha256=digest(args.run / "report.json"), review_sha256=digest(args.run / "review.json"),
                         records=records,graphs=report["graphs"],timings=report["timings"],
                         elapsed_seconds=report["elapsed_seconds"],neural_fits=0,candidate_promoted=False))
    if args.receipt:
        rows = []
        for row in report["records"]:
            if row["space"] != "correction" or row["library"] != 50000 or row["population"] != "development":
                continue
            group = row["score"]["groups"]["all"]
            rows.append({k:row[k] for k in ("seed","mode","k","shuffled")} |
                        {k:group[k] for k in ("components","pass_count","base_pass_count","repaired","damaged")})
        save(args.receipt,dict(schemaVersion=1,items=[dict(id="local-transfer",title="Do nearby states supply useful corrections?",
            queries=[dict(id="paired-diagnostic",source=dict(label="DFODE local-predictability diagnostic",
                files=[dict(label=args.output.name)],
                links=[dict(label="Research result review",url="https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/research/RESULTS.md")],
                metricDefinitions=[dict(label="Physical acceptance",definition="A signed increment passes when its absolute error is at most 1e-15 plus 0.1 times the reference magnitude.")],
                filters=["Population: 1,023 development states; 58 active species", "Library: 50,000 training states; 119 fixed features"],
                caveats=["Repeatedly inspected development data are not an independent test.","These are diagnostic transfers, not promoted models or proofs of intrinsic learnability."],
                evidenceFlow=[dict(kind="validation",title="Independent recount",detail="Separate code reproduced all 144 pass, base-pass, repair and damage records from saved physical predictions.")]),
                columns=["seed","mode","k","shuffled","components","pass_count","base_pass_count","repaired","damaged"],rows=rows,
                preview=dict(kind="aggregate",note="The fixed 50k/119-feature development comparison is shown; the linked summary retains all registered comparisons."),
                methods=[dict(language="calculation",code="acceptance = pass_count / components\nnet pass change = repaired - damaged")])])]))
    print(args.output)


if __name__ == "__main__":
    main()
