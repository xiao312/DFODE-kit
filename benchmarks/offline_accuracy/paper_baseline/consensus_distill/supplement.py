"""Close independent review checks without changing the archived primary review."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays, read_json, digest, checked_file
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);a=p.parse_args();root=a.run
    if (root/"supplement.json").exists():raise ValueError("Supplement exists")
    review=read_json(root/"review.json");report=read_json(checked_file(root/"report.json",review["report_sha256"]))
    for name,sha in report["artifacts"].items():checked_file(root/name,sha)
    records=[]
    for seed in (20261011,20261012):
        staged=root/"prepared"/str(seed);train=arrays(staged/"training-reference.npz");fit=arrays(staged/"fit-input/training.npz")
        assert set(fit)==set(train)|{"other"}
        for key,value in train.items():np.testing.assert_array_equal(value,fit[key])
        record=read_json(root/str(seed)/"result.json")
        assert record["teacher_pass_mask_differences"]==0
        assert record["teacher_formula_max_budget_difference"]<1e-12
        for population in ("training","same_source_unseen","development"):
            measured=read_json(root/str(seed)/population/"result.json")
            for item in measured["records"]:
                t=item["timings"];ratio=float(np.median(t["candidate"])/np.median(t["anchor"]))
                assert ratio==item["ratio"] and all(v>0 for values in t.values() for v in values)
            cells={r["arm"]:r for r in review["summary"] if r["seed"]==seed and r["population"]==population}
            records.append(dict(seed=seed,population=population,head_minus_constant=cells["head"]["acceptance"]-cells["constant"]["acceptance"]))
    save(root/"supplement.json",dict(status="verified",review_sha256=digest(root/"review.json"),head_vs_constant=records,
         hypothesis="Not supported: head fails promotion and does not consistently exceed constant on U/D.",
         checks="All staged fitting fields aligned; teacher reconstruction has zero pass flips and <1e-12 budget roundoff; timing ratios independently reconstructed."))
    print(digest(root/"review.json"));print(digest(root/"supplement.json"));print(records)


if __name__=="__main__":main()
