"""Trusted metadata; only states and frozen prediction assets enter worker."""
import shutil
import numpy as np
from ..kinetic_features.prepare import stage
from ..hidden_features.campaign import run_worker
from benchmarks.research_contract.adapter import arrays


def execute(project,dest,output,seed,state,gpu):
    model=dest/"projection-model";model.mkdir()
    for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(dest/"model"/name,model/name)
    active=arrays(model/"preprocessing.npz")["active"]
    matrix=np.load(dest/"element-matrix.npy")[:,active];matrix=matrix[np.any(matrix!=0,axis=1)]
    if np.linalg.matrix_rank(matrix)!=len(matrix):raise ValueError("Dependent nonzero element rows")
    np.save(model/"element-matrix.npy",matrix)
    return run_worker(project,dest/"inference-input",output,["--seed",str(seed)],state,gpu,model,
        worker_module="benchmarks.offline_accuracy.paper_baseline.element_projection.worker")
