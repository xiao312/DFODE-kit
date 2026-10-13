"""Bounded full-data optimization of an unchanged fixed local basis."""
import math
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.research_contract.adapter import arrays,digest
from ..hidden_features.prepare import save
from ..correction_scaling.model import fixed_batches
from ..decoder_loss.coordinates import decode
from .model import Basis,cached_coordinate


def objective(design,target,coefficient,ridge=1e-6):
    prediction=torch.bmm(design,coefficient[:,:,None]).squeeze(2)
    loss=torch.nn.functional.smooth_l1_loss(prediction,target,beta=math.log1p(.1))
    return loss+ridge*coefficient.square().mean()


def irls_step(design,target,coefficient,ridge=1e-6):
    """Species-major design [S,N,K]; exact convex quadratic majorizer solve."""
    n,k=design.shape[1:]
    residual=torch.bmm(design,coefficient[:,:,None]).squeeze(2)-target
    weight=residual.abs().clamp_min(math.log1p(.1)).reciprocal()
    weighted=design*weight[:,:,None]
    matrix=torch.bmm(design.transpose(1,2),weighted)/n
    matrix=matrix+(2*ridge/k)*torch.eye(k,dtype=design.dtype,device=design.device)
    rhs=torch.bmm(weighted.transpose(1,2),target[:,:,None])/n
    factor,info=torch.linalg.cholesky_ex(matrix)
    if info.any():raise ValueError("Non-positive normal matrix: no hidden jitter")
    solution=torch.cholesky_solve(rhs,factor).squeeze(2)
    if not torch.isfinite(solution).all():raise ValueError("Nonfinite coefficient solve")
    return solution


def fit_refine(rows,plan,destination):
    start=time.perf_counter();geometry=arrays("/input/geometry.npz")
    shutil.copyfile("/input/geometry.npz",destination/"geometry.npz")
    basis=Basis(geometry["centres"],geometry["weights"],geometry["bandwidth"]).cuda()
    initial=torch.load("/input/initial-weights.pt",weights_only=True,map_location="cuda")
    coefficient=initial.clone()
    with torch.no_grad():
        phi=torch.cat([basis(torch.as_tensor(x,device="cuda"))[:n] for n,x in fixed_batches(rows["features"])])
        design=phi.permute(1,0,2).double().contiguous();del phi
        target=torch.as_tensor(rows["target"].T.copy(),dtype=torch.float64,device="cuda")
        # Geometry rank is a training-only diagnostic, not a selection rule.
        gram=torch.bmm(design.transpose(1,2),design)/design.shape[1]
        eigenvalues=torch.linalg.eigvalsh(gram).cpu().numpy()
        np.savez(destination/"conditioning.npz",gram_eigenvalues=eigenvalues)
        history=[]
        for step in range(plan["updates"]+1):
            c=cached_coordinate(basis,coefficient,rows["features"])
            d,b=rows["delta"][:,rows["active"]],rows["base"][:,rows["active"]]
            prediction=decode(b,c,dict(decoder="asinh",scale_floor=1e-14))
            value=float(objective(design,target,coefficient.double()))
            derivative=(torch.bmm(design,coefficient.double()[:,:,None]).squeeze(2)-target)/math.log1p(.1)
            derivative=derivative.clamp(-1,1)
            gradient=torch.bmm(design.transpose(1,2),derivative[:,:,None]).squeeze(2)/len(rows["target"])
            gradient+=2e-6/coefficient.shape[1]*coefficient.double()
            item=dict(update=step,objective=value,acceptance=float((abs(prediction-d)<=1e-15+.1*abs(d)).mean()),
                      coordinate_mae=float(abs(c-rows["target"]).mean()),gradient_max=float(gradient.abs().max()),
                      elapsed_seconds=time.perf_counter()-start)
            history.append(item);save(destination/"progress.json",dict(history=history));print(item,flush=True)
            if step==plan["updates"]:break
            solution=irls_step(design,target,coefficient.double())
            coefficient=solution.float()
            after=float(objective(design,target,coefficient.double()))
            if after>value+1e-8*max(1,abs(value)):
                raise ValueError("Objective increased beyond FP32 roundoff allowance")
            item["solved_objective"]=float(objective(design,target,solution))
            item["rounded_objective"]=after
        torch.save(coefficient.cpu(),destination/"weights.pt")
        loaded=torch.load(destination/"weights.pt",weights_only=True,map_location="cuda")
        np.testing.assert_array_equal(c,cached_coordinate(basis,loaded,rows["features"]))
        output=rows["base"].copy();output[:,rows["active"]]=prediction
        np.savez(destination/"training-replay.npz",prediction=output,coordinate=c)
        return dict(history=history,parameter_count=coefficient.numel(),final_checkpoint=True,
                    total_seconds=time.perf_counter()-start,weights_sha256=digest(destination/"weights.pt"),
                    initial_weights_sha256=digest(Path("/input/initial-weights.pt")),geometry_sha256=digest(destination/"geometry.npz"))
