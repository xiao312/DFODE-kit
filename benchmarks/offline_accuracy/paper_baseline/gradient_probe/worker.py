"""Training-only finite-step probe. Never exports a modified model."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays, read_json
from benchmarks.offline_accuracy.paired.runtime import configure
from benchmarks.flame_conditioning.coordinates import input_features
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base, base_prediction
from ..adaptive.model import base_preprocessing
from ..adaptive.plan import base_config
from ..matched_targets.coordinates import differentiable_decode
from .metrics import geometry, physical_counts, groups


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True)
    args=p.parse_args();env=configure();start=time.perf_counter()
    plan=read_json("/input/config.json");rows=arrays("/input/training.npz")
    base,prep=load_base(Path("/model"),args.seed);bp=base_preprocessing(prep);active=prep["active"]
    original={k:v.clone() for k,v in base.state_dict().items()}
    for parameter in base.parameters():parameter.requires_grad_(True)
    params=list(base.parameters());trunk=sum(v.numel() for v in params[:-2])
    def tensor(value):return torch.as_tensor(value,dtype=torch.float64,device="cuda")
    x=torch.as_tensor((input_features(rows["states"])-bp["x_offset"])/bp["x_scale"],dtype=torch.float32,device="cuda")
    y=tensor(rows["delta"][:,active]);initial=tensor(rows["states"][:,2:][:,active])
    def loss(indices):
        prediction=differentiable_decode(initial[indices],base(x[indices]).double()*tensor(bp["y_scale"])+tensor(bp["y_offset"]),base_config(args.seed))
        values=torch.log1p(abs(prediction-y[indices])/(1e-15+.1*abs(y[indices])))
        if not torch.isfinite(values).all():raise ValueError("Nonfinite physical objective")
        return values,prediction
    batches=np.arange(len(x)).reshape(plan["batches"],plan["batch_size"])
    records=[];saved={"source_indices":rows["source_indices"],"truth":rows["delta"][:,active],"baseline":rows["base"][:,active]}
    for index,indices in enumerate(batches):
        base.load_state_dict(original);check=batches[(index+1)%len(batches)]
        masks=groups(saved["truth"][indices],saved["baseline"][indices]);names=list(masks)
        values,pred=loss(indices)
        diff=abs(pred.detach().cpu().numpy()-saved["baseline"][indices])/(1e-15+.1*abs(saved["truth"][indices]))
        deployed,_=base_prediction(base,bp,rows["states"][indices],args.seed)
        np.testing.assert_array_equal(deployed[:,active],saved["baseline"][indices])
        discrepancy=dict(maximum=float(diff.max()),p99=float(np.quantile(diff,.99)),
                         pass_disagreements=int(((values.detach().cpu().numpy()<=np.log(2)) != (abs(saved["baseline"][indices]-saved["truth"][indices])<=1e-15+.1*abs(saved["truth"][indices]))).sum()))
        if discrepancy["maximum"]>1e-5 or discrepancy["pass_disagreements"]:
            save(Path("/output/result.json"),dict(status="DECODER_MISMATCH",batch=index,discrepancy=discrepancy,environment=env))
            return
        gradients=[];means={}
        for name,mask in masks.items():
            means[name]=float(values[torch.as_tensor(mask,device="cuda")].mean()) if mask.any() else None
            if mask.any():
                grad=torch.autograd.grad(values[torch.as_tensor(mask,device="cuda")].mean(),params,retain_graph=True)
                gradients.append(torch.cat([v.flatten() for v in grad]).detach())
            else:gradients.append(torch.zeros(sum(v.numel() for v in params),device="cuda"))
        combined=gradients[1]*masks["base_pass"].mean()+gradients[2]*masks["base_fail"].mean()
        residual=float((combined-gradients[0]).norm()/gradients[0].norm().clamp_min(1e-30))
        if residual>1e-5:raise ValueError("Population gradient identity failed")
        geometry_by_block={name:geometry([g[sl] for g in gradients]) for name,sl in dict(all=slice(None),trunk=slice(None,trunk),final=slice(trunk,None)).items()}
        for rate in plan.get("learning_rates",[plan.get("learning_rate")]):
            base.load_state_dict(original)
            optimizer=torch.optim.Adam(params,lr=rate,foreach=False)
            optimizer.zero_grad();loss(indices)[0].mean().backward();optimizer.step()
            displacement=torch.cat([(p-original[k]).flatten() for (k,_),p in zip(base.named_parameters(),params)]).detach().double()
            record_id=str(index) if "learning_rates" not in plan else f"{index}_{rate}"
            record=dict(batch=index,record_id=record_id,learning_rate=rate,groups=names,geometry=geometry_by_block,gradient_identity_relative=residual,
                        discrepancy=discrepancy,step_norm=float(displacement.norm()),
                        first_order={name:float(g.double()@displacement) for name,g in zip(names,gradients)},before_loss=means,populations={})
            for label,selection in [("step",indices),("check",check)]:
                prediction,_=base_prediction(base,bp,rows["states"][selection],args.seed)
                pred=prediction[:,active];truth=saved["truth"][selection];old=saved["baseline"][selection]
                mask=groups(truth,old)
                record["populations"][label]=dict(before=physical_counts(old,truth,old,mask),after=physical_counts(pred,truth,old,mask))
                saved[f"prediction_{record_id}_{label}"]=pred
            records.append(record)
    base.load_state_dict(original)
    if not all(torch.equal(v,original[k]) for k,v in base.state_dict().items()):raise ValueError("Restoration failed")
    np.savez(Path("/output/predictions.npz"),**saved)
    save(Path("/output/result.json"),dict(status="complete",seed=args.seed,environment=env,records=records,
         restored=True,optimizer="fresh Adam; no historical moment state",seconds=time.perf_counter()-start))


if __name__=="__main__":main()
