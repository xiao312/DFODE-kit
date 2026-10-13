"""Input-only weighted least-change projection, with explicit rank fallback."""
import torch


def project(base, matrix, arm):
    if arm not in ("uniform","amplitude"):raise ValueError("Unknown geometry")
    b=base.double();a=matrix.double()
    if b.ndim!=2 or a.ndim!=2 or b.shape[1]!=a.shape[1]:raise ValueError("Axes differ")
    if not torch.isfinite(b).all() or not torch.isfinite(a).all():raise ValueError("Nonfinite projection input")
    scale=torch.ones_like(b) if arm=="uniform" else 1e-14+b.abs()
    scale=scale/scale.amax(dim=1,keepdim=True)
    u,s,vh=torch.linalg.svd(a[None,:,:]*scale[:,None,:],full_matrices=False)
    fallback=s[:,-1]<=1e-12*s[:,0]
    result=b.clone();ok=~fallback
    if ok.any():
        residual=b[ok]@a.T
        coefficient=torch.bmm(u[ok].transpose(1,2),residual[:,:,None]).squeeze(-1)/s[ok]
        correction=torch.bmm(vh[ok].transpose(1,2),coefficient[:,:,None]).squeeze(-1)*scale[ok]
        result[ok]=b[ok]-correction
    if not torch.isfinite(result).all():raise ValueError("Nonfinite projected increment")
    return result,fallback
