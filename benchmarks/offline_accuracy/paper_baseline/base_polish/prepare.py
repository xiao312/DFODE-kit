"""Reuse exact 50k rows and state-only evaluation inputs; no calibration refit."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save
from ..adaptive.plan import configuration


def stage(project,output,plan):
    parent=project/plan["parent_run"]
    review=read_json(checked_file(parent/"review.json",plan["parent_review_sha256"]))
    report=read_json(checked_file(parent/"report.json",review["report_sha256"]))
    for relative,expected in report["artifacts"].items():
        if relative.startswith("prepared/"):checked_file(parent/relative,expected)
    for seed in plan["seeds"]:
        source=parent/"prepared"/str(seed);dest=output/"prepared"/str(seed);dest.mkdir(parents=True)
        for name in ["model","input-training","input-same_source_unseen","input-development"]:shutil.copytree(source/name,dest/name)
        shutil.copyfile(source/"training-indices.npy",dest/"training-indices.npy")
        labels=arrays(source/"train-global/training.npz");rows=arrays(source/"input-training/states.npz")
        np.testing.assert_array_equal(labels["source_indices"],rows["source_indices"])
        for arm in plan["arms"]:
            path=dest/("train-"+arm);path.mkdir()
            np.savez(path/"training.npz",states=rows["states"],delta=labels["delta"],base=labels["base"],source_indices=rows["source_indices"])
            shutil.copytree(source/"model",path/"model")
            cfg=configuration("continue",seed)
            cfg.update(experiment=plan["experiment_id"],comparison_arm=arm,updates=plan["updates"],
                       batch_size=plan["batch_sizes"][arm],microbatch_size=1000,learning_rate=1e-7,final_learning_rate=1e-8,
                       diagnostics_every_updates=500,wall_seconds=1800,optimizer_foreach=False)
            save(path/"config.json",cfg)
        save(dest/"manifest.json",dict(training_states=50000,preprocessing="frozen parent",unseen_overlap=0,
             artifacts={str(p.relative_to(dest)):digest(p) for p in dest.rglob("*") if p.is_file()}))
