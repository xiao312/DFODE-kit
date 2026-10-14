"""Independent cross-fit recount and artifact-only inference replay."""
import argparse
from pathlib import Path
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);args=p.parse_args();root=args.run
    if (root/"review.json").exists():p.error("Review exists; do not overwrite")
    report=read_json(checked_file(root/"report.json",read_json(root/"verification.json")["report_sha256"]))
    for path,sha in report["artifacts"].items():checked_file(root/path,sha)
    plan=read_json(root/"frozen-experiment.json");table=[];evidence=[]
    for seed in plan["seeds"]:
        rows=arrays(root/str(seed)/"predictions.npz");original=arrays(root/"prepared"/str(seed)/"training.npz")
        result=read_json(root/str(seed)/"result.json");active=np.array(result["active_species_indices"])
        prep=arrays(root/"prepared"/str(seed)/"model/preprocessing.npz")
        np.testing.assert_array_equal(active,np.flatnonzero(prep["active"]))
        np.testing.assert_array_equal(rows["source_indices"],original["source_indices"])
        np.testing.assert_array_equal(rows["truth"],original["delta"][:,active]);np.testing.assert_array_equal(rows["baseline"],original["base"][:,active])
        np.testing.assert_array_equal(rows["initial"],original["states"][:,2:][:,active])
        n=len(rows["truth"]);fold=np.empty(n,dtype=int);fold[np.random.default_rng(seed+411).permutation(n)]=np.arange(n)%2
        np.testing.assert_array_equal(rows["fold"],fold);assert (np.bincount(fold)==n//2).all()
        d,b=rows["truth"],rows["baseline"];tau=1e-15+.1*abs(d);old=abs(b-d)<=tau;nt=abs(d)>1e-15/.9
        for arm in plan["arms"]:
            y=rows[arm];models=read_json(root/str(seed)/(arm+"-maps.json"))
            for held in range(2):
                fit=fold!=held;check=fold==held
                for i,model in enumerate(models[held]):
                    # Reconstruct fit-only edge selection independently, then use
                    # exact algebra for replay. Stable decoder may differ at ulps.
                    expected=np.unique(np.quantile(np.arcsinh(b[fit,i]/1e-14),np.arange(1,plan["bins"])/plan["bins"])) if arm=="conditional" else np.empty(0)
                    np.testing.assert_array_equal(expected,model["edges"])
                    bins=np.searchsorted(expected,np.arcsinh(b[fit,i]/1e-14),side="right")
                    np.testing.assert_array_equal(np.bincount(bins,minlength=len(expected)+1),model["counts"])
                    selected=np.searchsorted(expected,np.arcsinh(b[check,i]/1e-14),side="right")
                    c=np.array(model["constants"])[selected]
                    decoded=1e-14*np.sinh(np.arcsinh(b[check,i]/1e-14)+c)
                    np.testing.assert_allclose(decoded,y[check,i],rtol=1e-13,atol=1e-28)
                    # Counts must still match using the independent formula.
                    assert np.array_equal(abs(decoded-d[check,i])<=tau[check,i],abs(y[check,i]-d[check,i])<=tau[check,i])
            assert np.isfinite(y).all();passed=abs(y-d)<=tau
            expected=next(r for r in result["records"] if r["arm"]==arm)["crossfit"]
            masks=dict(all=np.ones_like(nt),base_pass=old,base_fail=~old,floor=~nt,weak_nontrivial=nt&(abs(d)<=1e-12),active_increment=abs(d)>1e-12)
            for name,mask in masks.items():
                g=expected["groups"][name]
                actual=[int(mask.sum()),int((passed&mask).sum()),int((passed&~old&mask).sum()),int((~passed&old&mask).sum())]
                assert actual==[g[k] for k in ("components","passes","repaired","damaged")]
            signs=int(((np.sign(y)!=np.sign(d))&nt).sum());negative=int(((rows["initial"]+y)<0).sum())
            assert signs==expected["sign_errors"] and negative==expected["negative"]
            np.testing.assert_array_equal(passed.sum(0),expected["species_passes"])
            np.testing.assert_allclose(np.quantile(abs(y-d)/tau,.99),expected["p99"],rtol=1e-14)
            repaired=int((passed&~old).sum());damaged=int((~passed&old).sum());R=repaired/(~old).sum();D=damaged/old.sum()
            assert abs(passed.mean()-(old.mean()+(1-old.mean())*R-old.mean()*D))<3e-16
            table.append(dict(seed=seed,arm=arm,acceptance=float(passed.mean()),base=float(old.mean()),repaired=repaired,damaged=damaged,damage_rate=float(D),nontrivial=float(passed[nt].mean()),base_nontrivial=float(old[nt].mean()),sign_errors=signs,base_sign_errors=int(((np.sign(b)!=np.sign(d))&nt).sum()),negative=negative,base_negative=int(((rows["initial"]+b)<0).sum()),p99=expected["p99"]))
        a,c=[next(t for t in table if t["seed"]==seed and t["arm"]==arm) for arm in ("conditional","constant")]
        evidence.append(a["acceptance"]>=max(a["base"],c["acceptance"])+.005 and a["nontrivial"]>=a["base_nontrivial"] and a["damage_rate"]<=.02 and a["sign_errors"]<=a["base_sign_errors"] and a["negative"]<=a["base_negative"])
    save(root/"review.json",dict(status="verified",report_sha256=digest(root/"report.json"),table=table,advance=all(evidence),candidate=False,confirmation_opened=False))
    print(root/"review.json")


if __name__=="__main__":main()
