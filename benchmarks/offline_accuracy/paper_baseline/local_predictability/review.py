"""Independently recount diagnostic transfers from saved physical arrays."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, checked_file, digest, project_path
from ..hidden_features.prepare import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new review file")
    report = read_json(checked_file(args.run / "report.json",read_json(args.run / "verification.json")["report_sha256"]))
    for path,value in report["artifacts"].items():
        checked_file(project_path(args.run,path),value)
    binding = read_json(args.project / "research/bindings.json")
    active = np.array([s != "AR" for s in binding["species"]])
    reviewed = []
    for record in report["records"]:
        root = args.run / str(record["seed"])
        query, pred = arrays(root / "queries.npz"), arrays(root / record["artifact"])
        np.testing.assert_array_equal(query["source_indices"],pred["source_indices"])
        np.testing.assert_array_equal(query["population"],pred["population"])
        mask = query["population"] == record["population"]
        d,b,y = (a[mask][:,active] for a in (query["delta"],query["base"],pred["prediction"]))
        allowance = 1e-15+.1*abs(d)
        old,passed = abs(b-d)<=allowance,abs(y-d)<=allowance
        expected = record["score"]["groups"]["all"]
        actual = dict(components=int(passed.size),pass_count=int(passed.sum()),
                      base_pass_count=int(old.sum()),repaired=int((~old & passed).sum()),damaged=int((old & ~passed).sum()))
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("Independent physical count differs")
        if not np.isfinite(pred["prediction"]).all():
            raise ValueError("Nonfinite diagnostic output")
        reviewed.append({k:record[k] for k in ("seed","space","library","k","mode","shuffled","population")} | actual)
    save(args.output,dict(status="verified",report_sha256=digest(args.run / "report.json"),
                         records=reviewed,candidate_promoted=False,independent_confirmation=False))
    print(args.output)


if __name__ == "__main__":
    main()
