"""Reuse exact 50k rows and state-only evaluation inputs; no calibration refit."""
import shutil
import numpy as np
from benchmarks.research_contract.adapter import arrays,read_json,checked_file,digest
from ..hidden_features.prepare import save
from ..adaptive.plan import configuration


def expanded_indices(total,prefix,unseen):
    """Keep the old prefix and exclude the fixed evaluation positions exactly."""
    prefix=np.asarray(prefix,dtype=np.int64);unseen=np.asarray(unseen,dtype=np.int64)
    for values in (prefix,unseen):
        if len(np.unique(values))!=len(values) or np.any(values<0) or np.any(values>=total):raise ValueError("Invalid row positions")
    if np.intersect1d(prefix,unseen).size:raise ValueError("Evaluation row in training prefix")
    rest=np.setdiff1d(np.arange(total),np.r_[prefix,unseen])
    selected=np.r_[prefix,rest]
    if len(selected)!=total-len(unseen):raise ValueError("Selection count differs")
    return selected


def stage(project,output,plan):
    parent=project/plan["parent_run"]
    review=read_json(checked_file(parent/"review.json",plan["parent_review_sha256"]))
    report=read_json(checked_file(parent/"report.json",review["report_sha256"]))
    if plan.get("control_run"):
        control=project/plan["control_run"]
        control_review=read_json(checked_file(control/"review.json",plan["control_review_sha256"]))
        checked_file(control/"report.json",control_review["report_sha256"])
    for relative,expected in report["artifacts"].items():
        if relative.startswith("prepared/"):checked_file(parent/relative,expected)
    for seed in plan["seeds"]:
        source=parent/"prepared"/str(seed);dest=output/"prepared"/str(seed);dest.mkdir(parents=True)
        for name in ["model","input-same_source_unseen","input-development"]:shutil.copytree(source/name,dest/name)
        indices=np.load(source/"training-indices.npy",allow_pickle=False)
        labels=arrays(source/"train-global/training.npz");rows=arrays(source/"input-training/states.npz")
        np.testing.assert_array_equal(labels["source_indices"],rows["source_indices"])
        if plan["training_count"]!=50000:
            binding=read_json(project/"research/bindings.json")
            full=arrays(project/binding["prepared_root"]/str(seed)/"training.npz")
            unseen=arrays(project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
            indices=expanded_indices(len(full["states"]),indices,unseen)
            new={key:full[key][indices] for key in ("states","delta","base","source_indices")}
            for key in ("delta","base","source_indices"):np.testing.assert_array_equal(new[key][:50000],labels[key])
            np.testing.assert_array_equal(new["states"][:50000],rows["states"])
            rows=labels=new
            del full
        unseen_ids=arrays(dest/"input-same_source_unseen/states.npz")["source_indices"]
        if np.isin(rows["source_indices"],unseen_ids).any() or len(rows["states"])!=plan["training_count"]:raise ValueError("Population exposure differs")
        np.save(dest/"training-indices.npy",indices)
        (dest/"input-training").mkdir()
        np.savez(dest/"input-training/states.npz",states=rows["states"],source_indices=rows["source_indices"])
        for arm in plan["arms"]:
            path=dest/("train-"+arm);path.mkdir()
            np.savez(path/"training.npz",states=rows["states"],delta=labels["delta"],base=labels["base"],source_indices=rows["source_indices"])
            shutil.copytree(source/"model",path/"model")
            cfg=configuration("continue",seed)
            cfg.update(experiment=plan["experiment_id"],comparison_arm=arm,updates=plan["updates"],
                       batch_size=plan["batch_sizes"][arm],microbatch_size=1000,learning_rate=1e-7,final_learning_rate=1e-8,
                       diagnostics_every_updates=500,wall_seconds=1800,optimizer_foreach=False)
            save(path/"config.json",cfg)
        save(dest/"manifest.json",dict(training_states=len(rows["states"]),preprocessing="frozen parent",unseen_overlap=0,
             artifacts={str(p.relative_to(dest)):digest(p) for p in dest.rglob("*") if p.is_file()}))
