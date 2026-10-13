"""Fit gate only; both chemistry predictors and all evaluation roles stay fixed."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,read_json,digest
from benchmarks.offline_accuracy.paired.runtime import configure
from ..hidden_features.prepare import save
from ..correction_scaling.model import load_base,load_head,prediction,base_prediction
from ..correction_scaling.plan import configuration
from ..adaptive.model import base_preprocessing,features
from ..conditional_calibration.worker import summary
from .model import make_gate,replace,calibrate


def probabilities(model,x,c):
    with torch.no_grad():
        return torch.cat([model(x[start:start+1024],c[start:start+1024]).double().sigmoid().cpu() for start in range(0,len(x),1024)]).numpy()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--seed",type=int,required=True);args=p.parse_args()
    env=configure();started=time.monotonic();plan=read_json("/input/config.json");rows=arrays("/input/training.npz")
    cfg=configuration(args.seed,10000,"coordinate-sl1");base,prep=load_base(Path("/model"),args.seed);head=load_head(Path("/model"),prep,cfg)
    frozen,_=base_prediction(base,base_preprocessing(prep),rows["states"],args.seed)
    np.testing.assert_array_equal(frozen,rows["base"])
    raw,coordinate=prediction(base,head,prep,rows["states"],cfg)
    replay,rc=prediction(base,head,prep,rows["states"],cfg)
    np.testing.assert_array_equal(raw,replay);np.testing.assert_array_equal(coordinate,rc)
    xvalues=features(rows["states"],frozen,prep,cfg).astype(np.float32)
    np.testing.assert_array_equal(xvalues,rows["features"])
    active=prep["active"];b=frozen[:,active];d=rows["delta"][:,active];candidate=raw[:,active];initial=rows["states"][:,2:][:,active]
    fit_end=plan["fit_rows"];cal_end=fit_end+plan["calibration_rows"]
    if len(d)!=cal_end+plan["check_rows"]:raise ValueError("Role sizes differ")
    roles=dict(fitting=slice(0,fit_end),calibration=slice(fit_end,cal_end),check=slice(cal_end,None))
    scale=np.maximum(np.sqrt(np.mean(coordinate[:fit_end]**2,axis=0)),1e-6)
    x=torch.as_tensor(xvalues,dtype=torch.float32,device="cuda");c=torch.as_tensor(coordinate/scale,dtype=torch.float32,device="cuda")
    old=abs(b-d)<=1e-15+.1*abs(d);new=abs(candidate-d)<=1e-15+.1*abs(d)
    mask=torch.as_tensor((old!=new)[:fit_end],device="cuda");labels=torch.as_tensor((~old&new)[:fit_end],dtype=torch.float64,device="cuda")
    saved=dict(source_indices=rows["source_indices"],baseline=b,truth=d,initial=initial,raw=candidate,coordinate=coordinate,features=xvalues,scale=scale)
    records=[]
    for arm in plan["arms"]:
        model=make_gate(x.shape[1],d.shape[1],arm,args.seed).to("cuda");optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=1e-3,foreach=False)
        rng=np.random.default_rng(args.seed+907);history=[]
        for step in range(plan["batches"]):
            if time.monotonic()-started>plan["resources"]["maximum_worker_seconds"]:raise TimeoutError("Registered worker budget")
            if step%(fit_end//plan["batch_size"])==0:order=rng.permutation(fit_end)
            start=(step%(fit_end//plan["batch_size"]))*plan["batch_size"]
            ix=torch.as_tensor(order[start:start+plan["batch_size"]],device="cuda")
            logits=model(x[ix],c[ix]).double();selected=mask[ix]
            if not selected.any():raise ValueError("No discordant fitting components")
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[selected],labels[ix][selected])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.isfinite(loss) or any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):raise ValueError("Nonfinite gate fit")
            optimizer.param_groups[0]["lr"]=1e-4+.5*(1e-3-1e-4)*(1+np.cos(np.pi*step/(plan["batches"]-1)))
            optimizer.step()
            if (step+1)%500==0:history.append(dict(update=step+1,loss=float(loss),seconds=time.monotonic()-started))
        torch.save(model.state_dict(),Path("/output")/(arm+".pt"))
        np.savez(Path("/output")/(arm+".npz"),**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})
        prob=probabilities(model,x,c)
        again=make_gate(x.shape[1],d.shape[1],arm,args.seed).to("cuda");again.load_state_dict(torch.load(Path("/output")/(arm+".pt"),weights_only=True,map_location="cuda"))
        np.testing.assert_array_equal(prob,probabilities(again,x,c))
        sl=roles["calibration"];threshold,curve=calibrate(prob[sl],b[sl],candidate[sl],initial[sl],d[sl])
        pred,chosen=replace(b,candidate,initial,prob,threshold)
        saved[arm]=pred;saved[arm+"_probability"]=prob;saved[arm+"_selected"]=chosen
        record=dict(arm=arm,threshold=threshold,calibration_curve=curve,history=history,
                    trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                    populations={name:summary(pred[sl],d[sl],b[sl],initial[sl]) for name,sl in roles.items()})
        records.append(record)
    np.savez(Path("/output/predictions.npz"),**saved)
    save(Path("/output/result.json"),dict(status="complete",records=records,environment=env,
         base={name:summary(b[sl],d[sl],b[sl],initial[sl]) for name,sl in roles.items()},
         raw={name:summary(candidate[sl],d[sl],b[sl],initial[sl]) for name,sl in roles.items()},
         guard_blocked={name:int(((initial[sl]+candidate[sl])<0).sum()) for name,sl in roles.items()},
         prediction_sha256=digest("/output/predictions.npz"),active_species_indices=np.flatnonzero(active).tolist(),
         exact_replay=True,seconds=time.monotonic()-started,candidate=False))


if __name__=="__main__":main()
