"""Fixed selections made before candidate scoring or new reference integration."""
from collections import defaultdict
import numpy as np
from ..correction_scaling.coverage import magnitude_bins,LABELS

SELECTION_SEED = 20261013


def unseen_indices(snapshots,start=10000,per_snapshot=5000):
    rng = np.random.default_rng(SELECTION_SEED)
    selected = []
    for snapshot in np.unique(snapshots):
        candidates = np.flatnonzero((snapshots==snapshot)&(np.arange(len(snapshots))>=start))
        if len(candidates)<per_snapshot:
            raise ValueError("Insufficient correction-unseen snapshot support")
        selected.extend(rng.choice(candidates,per_snapshot,replace=False).tolist())
    return np.asarray(sorted(selected),dtype=int)


def audit_selection(tails,training,species,per_snapshot=16,control_stride=4):
    rng = np.random.default_rng(SELECTION_SEED)
    candidates = [dict(record,seed=seed) for seed,records in tails.items() for record in records]
    forbidden = {r["source_row"] for r in candidates}
    selected,used = [],set()
    snapshots = sorted({r["snapshot"] for r in candidates})
    for snapshot in snapshots:
        groups = defaultdict(list)
        for record in candidates:
            if record["snapshot"]==snapshot:
                groups[(record["seed"],record["species"],record["reference_bin"])].append(record)
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            groups[key].sort(key=lambda r:r["source_row"])
            rng.shuffle(groups[key])
        picked = 0
        while picked<per_snapshot:
            progress = False
            for key in keys:
                while groups[key] and groups[key][-1]["source_row"] in used:
                    groups[key].pop()
                if not groups[key]:
                    continue
                record = dict(groups[key].pop(),role="tail")
                selected.append(record); used.add(record["source_row"])
                picked += 1; progress = True
                if picked==per_snapshot:
                    break
            if not progress:
                raise ValueError("Insufficient distinct tail states")
    anchors = list(selected)
    for anchor in anchors[::control_stride]:
        data = training[anchor["seed"]]
        j = species.index(anchor["species"])
        b,d = data["base"][:,j],data["delta"][:,j]
        eligible = ((data["snapshot"]==anchor["snapshot"]) & (np.sign(b)*np.sign(d)>0)
            & (magnitude_bins(d)==LABELS.index(anchor["reference_bin"]))
            & ~np.isin(data["source_indices"],list(forbidden|used)))
        indices = np.flatnonzero(eligible)
        if not len(indices):
            raise ValueError("No same-sign control with matching snapshot/species/magnitude")
        # Tie break by source identity rather than seed-dependent prefix order.
        order = np.lexsort((data["source_indices"][indices],abs(data["states"][indices,0]-anchor["temperature_K"])))
        row = int(indices[order[0]])
        source = int(data["source_indices"][row])
        selected.append(dict(role="control",seed=anchor["seed"],source_row=source,
            training_row=row,species=anchor["species"],snapshot=anchor["snapshot"],
            reference_bin=anchor["reference_bin"],matched_tail_source_row=anchor["source_row"],
            temperature_gap_K=float(abs(data["states"][row,0]-anchor["temperature_K"])),
            pressure_gap_Pa=float(abs(data["states"][row,1]-anchor["pressure_Pa"]))))
        used.add(source)
    return selected
