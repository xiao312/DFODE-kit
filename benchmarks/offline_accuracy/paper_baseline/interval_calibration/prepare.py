"""Reuse the pinned 50k population without evaluation-label exposure."""
import shutil
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save
import numpy as np


def stage(project,output,plan):
    parent=project/plan["parent_run"]
    review=read_json(checked_file(parent/"review.json",plan["parent_review_sha256"]))
    report=read_json(checked_file(parent/"report.json",review["report_sha256"]))
    for relative,expected in report["artifacts"].items():
        if relative.startswith("prepared/"):checked_file(parent/relative,expected)
    for seed in plan["seeds"]:
        source=parent/"prepared"/str(seed);dest=output/"prepared"/str(seed);dest.mkdir(parents=True)
        for name in ["model","input-training","input-same_source_unseen","input-development"]:
            shutil.copytree(source/name,dest/name)
        shutil.copyfile(source/"training-indices.npy",dest/"training-indices.npy")
        rows=arrays(source/"train-global/training.npz")
        for arm in plan["arms"]:
            path=dest/("train-"+arm);path.mkdir()
            np.savez(path/"training.npz",**{k:rows[k] for k in ("delta","base","target","active","source_indices")})
            save(path/"config.json",dict(plan,seed=seed,arm=arm))
        save(dest/"manifest.json",dict(training_states=50000,unseen_overlap=0,
             artifacts={str(p.relative_to(dest)):digest(p) for p in dest.rglob("*") if p.is_file()}))
