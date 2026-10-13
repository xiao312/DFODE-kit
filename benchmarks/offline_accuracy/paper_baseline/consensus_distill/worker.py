"""Existing optimizer loop with a bounded teacher-fraction objective."""
import argparse
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..correction_scaling.plan import configuration
from ..correction_scaling.model import load_base
from ..correction_scaling.train import fit
from ..resolved_fit.fit import head
from ..hidden_features.model import base_and_hidden
from ..hidden_features.prepare import save
from ..paired_base.metrics import bounded_mean,strict_shrink
from .model import teacher_fraction,decode,objective,cached_prediction,initial_check


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();plan=read_json("/input/config.json");rows=arrays("/input/training.npz")
    if len(rows["states"])!=plan["fit_rows"]:raise ValueError("Wrong training population")
    base,prep=load_base(Path("/model"),args.seed);active=prep["active"]
    b,h=base_and_hidden(base,prep,rows["states"],args.seed);np.testing.assert_array_equal(b,rows["base"])
    target=teacher_fraction(b[:,active],rows["other"][:,active]);constant=target.mean(0)
    teacher=decode(b[:,active],target);original=strict_shrink(b[:,active],bounded_mean(b[:,active],rows["other"][:,active]))
    difference=abs(teacher-original)/(1e-15+.1*abs(rows["delta"][:,active]))
    training=dict(features=h,base=b,delta=rows["delta"],target=target)
    cfg=configuration(args.seed,10000,"coordinate-sl1")
    cfg.update(experiment=plan["experiment_id"],fit_objective="teacher-soft-bce",training_count=len(h),widths=plan["widths"],
        updates=plan["batches"],batch_size=plan["batch_size"],learning_rate=plan["learning_rate"],final_learning_rate=plan["final_learning_rate"],wall_seconds=plan["resources"]["maximum_fit_seconds"])
    root=Path("/output");save(root/"config.json",cfg);np.save(root/"constant.npy",constant)
    audit_ix=np.linspace(0,len(h)-1,32,dtype=int)
    np.savez(root/"hidden-audit.npz",indices=audit_ix,features=h[audit_ix])
    np.savez(root/"teacher.npz",fraction=target,source_indices=rows["source_indices"])
    model,record=fit(training,None,active,cfg,root,objective_function=objective,prediction_function=cached_prediction,initial_prediction_check=initial_check)
    if record["parameter_count"]!=plan["parameters"]:raise ValueError("Parameter count")
    if any(p.grad is not None or p.requires_grad for p in base.parameters()):raise ValueError("Frozen base gradient")
    restored=head(cfg,800,int(active.sum()));restored.load_state_dict(torch.load(root/"weights.pt",weights_only=True,map_location="cuda"))
    prediction,alpha=cached_prediction(model,training,active,cfg);again,alpha2=cached_prediction(restored,training,active,cfg)
    np.testing.assert_array_equal(prediction,again);np.testing.assert_array_equal(alpha,alpha2)
    np.savez(root/"training-replay.npz",prediction=prediction,fraction=alpha,source_indices=rows["source_indices"])
    save(root/"result.json",dict(status="complete",record=record,environment=env,exact_reload=True,
         teacher_formula_max_budget_difference=float(difference.max()),teacher_pass_mask_differences=int(np.count_nonzero((abs(teacher-rows["delta"][:,active])<=1e-15+.1*abs(rows["delta"][:,active]))!=(abs(original-rows["delta"][:,active])<=1e-15+.1*abs(rows["delta"][:,active])))),
         teacher_sha256=digest(root/"teacher.npz")))


if __name__=="__main__":main()
