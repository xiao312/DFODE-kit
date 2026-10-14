"""GPU fit or isolated state-only replay of a fixed-centre model."""
import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base,base_prediction,fixed_batches
from ..adaptive.model import base_preprocessing
from ..decoder_loss.coordinates import decode
from .model import Basis,geometry,cached_coordinate,predict


def fit(rows,plan,destination):
    start=time.perf_counter()
    bandwidth,fallback=geometry(rows["features"],rows["centre_indices"],rows["weights"])
    geom=dict(centres=rows["features"][rows["centre_indices"]],weights=rows["weights"],bandwidth=bandwidth,
              centre_ids=rows["source_indices"][rows["centre_indices"]])
    np.savez(destination/"geometry.npz",**geom)
    basis=Basis(geom["centres"],geom["weights"],bandwidth).cuda()
    with torch.no_grad():
        phi=torch.cat([basis(torch.as_tensor(x,device="cuda"))[:n] for n,x in fixed_batches(rows["features"])])
    target=torch.as_tensor(rows["target"],dtype=torch.float64,device="cuda")
    coefficient=torch.nn.Parameter(torch.zeros((58,plan["centres"]),device="cuda"))
    opt=torch.optim.Adam([coefficient],lr=.01)
    rng=torch.Generator(device="cuda").manual_seed(plan["seed"])
    prep_seconds=time.perf_counter()-start
    history=[]
    for step in range(plan["updates"]+1):
        if step % 1000 == 0 or step == plan["updates"]:
            c=cached_coordinate(basis,coefficient,rows["features"])
            d,b=rows["delta"][:,rows["active"]],rows["base"][:,rows["active"]]
            predicted=decode(b,c,dict(decoder="asinh",scale_floor=1e-14))
            report=dict(update=step,acceptance=float((abs(predicted-d)<=1e-15+.1*abs(d)).mean()),
                        coordinate_mae=float(abs(c-rows["target"]).mean()),elapsed_seconds=time.perf_counter()-start)
            history.append(report);print(report,flush=True)
            save(destination/"progress.json",dict(history=history))
        if step==plan["updates"]:
            break
        selected=torch.randint(len(target),(plan["batch_size"],),generator=rng,device="cuda")
        prediction=torch.einsum("nsk,sk->ns",phi[selected],coefficient)
        loss=torch.nn.functional.smooth_l1_loss(prediction.double(),target[selected],beta=math.log1p(.1))
        loss=loss+1e-6*coefficient.square().mean()
        opt.zero_grad();loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(coefficient.grad).all():
            raise ValueError("Nonfinite fit; no fallback")
        lr=.0001+.5*(.01-.0001)*(1+math.cos(math.pi*step/plan["updates"]))
        for group in opt.param_groups:group["lr"]=lr
        opt.step()
    torch.save(coefficient.detach().cpu(),destination/"weights.pt")
    torch.save(opt.state_dict(),destination/"optimizer.pt")
    c=cached_coordinate(basis,coefficient,rows["features"])
    reloaded=torch.load(destination/"weights.pt",weights_only=True,map_location="cuda")
    np.testing.assert_array_equal(c,cached_coordinate(basis,reloaded,rows["features"]))
    output=rows["base"].copy()
    output[:,rows["active"]]=decode(output[:,rows["active"]],c,dict(decoder="asinh",scale_floor=1e-14))
    np.savez(destination/"training-replay.npz",prediction=output,coordinate=c)
    return dict(history=history,parameter_count=coefficient.numel(),preparation_seconds=prep_seconds,
                bandwidth_fallback=fallback,total_seconds=time.perf_counter()-start,
                final_checkpoint=True,weights_sha256=digest(destination/"weights.pt"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["train","infer"])
    p.add_argument("--seed",required=True,type=int)
    args=p.parse_args()
    env=configure();destination=Path("/output")
    if args.mode=="train":
        plan=read_json("/input/config.json")
        if plan["seed"]!=args.seed:raise ValueError("Seed mismatch")
        if plan.get("optimizer_kind")=="irls":
            from .convex import fit_refine
            result=fit_refine(arrays("/input/training.npz"),plan,destination)
        else:
            result=fit(arrays("/input/training.npz"),plan,destination)
        save(destination/"result.json",dict(status="complete",environment=env,config=plan,**result))
        return
    rows=arrays("/input/states.npz")
    if set(rows)!={"states","source_indices"}:raise ValueError("State-only inference required")
    base,prep=load_base(Path("/model"),args.seed)
    geom=arrays("/model/geometry.npz")
    basis=Basis(geom["centres"],geom["weights"],geom["bandwidth"]).cuda()
    coefficients=torch.load("/model/weights.pt",weights_only=True,map_location="cuda")
    call=lambda values:predict(base,prep,basis,coefficients,values,args.seed)
    pred,c=call(rows["states"])
    again,ac=call(rows["states"])
    np.testing.assert_array_equal(pred,again);np.testing.assert_array_equal(c,ac)
    np.savez(destination/"predictions.npz",prediction=pred,coordinate=c,source_indices=rows["source_indices"])
    sample=rows["states"][:1023]
    calls=dict(candidate=lambda:call(sample),anchor=lambda:base_prediction(base,base_preprocessing(prep),sample,args.seed))
    times={name:[] for name in calls}
    for function in calls.values():function()
    for repeat in range(10):
        for name in (["candidate","anchor"] if repeat%2==0 else ["anchor","candidate"]):
            torch.cuda.synchronize();start=time.perf_counter();calls[name]();torch.cuda.synchronize()
            times[name].append(time.perf_counter()-start)
    save(destination/"result.json",dict(status="complete",environment=env,exact_repeat=True,
         states=len(pred),timing_states=len(sample),timings=times,prediction_sha256=digest(destination/"predictions.npz")))


if __name__=="__main__":main()
