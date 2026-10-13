"""Stage correction-unseen training covariates and pinned historical weights."""
import shutil
import numpy as np
import cantera as ct
from benchmarks.research_contract.adapter import arrays,read_json,checked_file
from ..hidden_features.prepare import save
from benchmarks.flame_conditioning.chemistry import element_matrix


def stage(project,dest,seed,plan):
    binding=read_json(project/"research/bindings.json");source=project/binding["prepared_root"]/str(seed)
    full=arrays(source/"training.npz")
    unseen=arrays(project/binding["transfer_root"]/"plan/selections.npz")[f"unseen_{seed}"]
    allowed=np.setdiff1d(np.arange(len(full["states"])),np.r_[np.arange(10000),unseen])
    chosen=np.random.default_rng(seed+plan["selection_seed_offset"]).permutation(allowed)[:50000]
    if len(chosen)!=50000 or len(np.unique(chosen))!=50000 or np.isin(chosen,unseen).any() or (chosen<10000).any():raise ValueError("Invalid role selection")
    payload={k:full[k][chosen] for k in ("states","delta","base","features","source_indices","snapshot")}
    np.savez(dest/"training.npz",**payload);np.save(dest/"selected-positions.npy",chosen)
    model=dest/"model";model.mkdir()
    for name in ("base-weights.pt","preprocessing.npz"):shutil.copyfile(source/name,model/name)
    head=binding["seeds"][str(seed)]["historical"]["coordinate-sl1"]
    directory=project/head["directory"];result=read_json(checked_file(directory/"result.json",head["result_sha256"]))
    shutil.copyfile(checked_file(directory/"weights.pt",result["artifacts"]["weights.pt"]),model/"weights.pt")
    save(dest/"config.json",plan)
    mechanism=checked_file(project/"runs/offline-accuracy/fuel-recipe-200k-20261009/dataset/mechanism.yaml",binding["mechanism_sha256"])
    gas=ct.Solution(str(mechanism))
    if gas.species_names!=binding["species"]:raise ValueError("Mechanism species axis differs")
    np.save(dest/"element-matrix.npy",element_matrix(gas))
    save(dest/"exposure.json",dict(correction_fit_count=10000,excluded_unseen_count=len(unseen),
         role_counts=[plan[k] for k in ("fit_rows","calibration_rows","check_rows")],base_seen=True,confirmation=False,
         species=binding["species"],head_result_sha256=head["result_sha256"]))
