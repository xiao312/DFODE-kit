"""Independent pass-bit reconstruction of reliability-table decisions."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project",type=Path,required=True);a=p.parse_args()
    root=a.project/"runs/offline-accuracy/pair-reliability-20261013";review=read_json(root/"review.json")
    if (root/"independent-check.json").exists():raise ValueError("Check exists")
    for name,sha in review["artifacts"].items():checked_file(root/name,sha)
    plan=read_json(root/"frozen-experiment.json");source=a.project/plan["source_run"]
    records=[]
    for seed in plan["seeds"]:
        x=arrays(source/"prepared"/str(seed)/"fit-input/training.npz");saved=arrays(root/f"{seed}.npz")
        np.testing.assert_array_equal(saved["source_indices"],x["source_indices"])
        b,q,d=x["base"],x["other"],x["delta"];budget=1e-15+.1*np.abs(d)
        before=np.less_equal(np.abs(b-d),budget);after=np.less_equal(np.abs(q-d),budget)
        eligible=(b*q>0)&(abs(q)<abs(b))
        # Use sign equality rather than product for any possible product underflow.
        eligible=(np.sign(b)==np.sign(q))&(q!=0)&(abs(q)<abs(b))
        bins=np.full(b.shape,-1,dtype=int);lr=np.log(abs(b[eligible]))-np.log(abs(q[eligible]))
        bins[eligible]=(lr>=np.log(1.1)).astype(int)+(lr>=np.log(2.)).astype(int)+(lr>=np.log(10.)).astype(int)
        reference_fit=read_json(root/f"{seed}-fit.json")["records"]
        expected=b.copy()
        for fold in range(2):
            train=saved["folds"]!=fold;check=~train;table=saved["tables"][fold]
            for species in range(b.shape[1]):
                for bucket in range(4):
                    mask=train&(bins[:,species]==bucket)
                    category=2*before[:,species][mask].astype(int)+after[:,species][mask].astype(int)
                    counts=np.bincount(category,minlength=4);bp=int(counts[2]+counts[3]);repair=int(counts[1]);damage=int(counts[2])
                    nt=mask&(abs(d[:,species])>1e-15/.9)
                    ntchange=int(after[nt,species].sum()-before[nt,species].sum())
                    selected=mask.sum()>=128 and bp>0 and repair>damage and damage<=.02*bp and ntchange>=0
                    assert bool(table[species,bucket])==bool(selected)
                    r=reference_fit[fold][species*4+bucket]
                    assert (r["count"],r["basepass"],r["repairs"],r["damage"],r["nontrivial_change"])==(int(mask.sum()),bp,repair,damage,ntchange)
                    apply=check&(bins[:,species]==bucket)&selected;expected[apply,species]=q[apply,species]
            records.append(dict(seed=seed,fold=fold,selected=int(table.sum())))
        np.testing.assert_array_equal(expected,saved["prediction"])
    save(root/"independent-check.json",dict(status="verified",review_sha256=digest(root/"review.json"),records=records,proceed_full_screen=review["proceed_full_screen"]))
    print(digest(root/"review.json"));print(digest(root/"independent-check.json"));print(records)


if __name__=="__main__":main()
