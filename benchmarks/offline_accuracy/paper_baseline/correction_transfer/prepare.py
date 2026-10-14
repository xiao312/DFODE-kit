"""Freeze identities and selections without inspecting candidate errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from ..matched_work.run import checked_result
from ..correction_scaling.run import prepared_manifest,load_arrays
from ..correction_scaling.plan import SEEDS,configuration
from .selection import unseen_indices,audit_selection

HEAD_HASHES = {
    20261011:"d8bb3b52e68ccc9d6e73e6521d7814ea8b90642fe757e829ef30228160bca5e9",
    20261012:"5f2bdec7fb505e73b0520023ef95f0df5297e17cbbfc1ecbdd63d6b0f1ed1c62",
}
GAMMAS = (0.,.125,.25,.5,.75,1.)


def inputs(original,candidate,dataset=None):
    manifests = {}
    for seed in SEEDS:
        manifests[seed] = prepared_manifest(original/"prepared",seed)
        head = candidate/"10000"/f"{seed}-coordinate-sl1"
        result = checked_result(head)
        if sha256(head/"result.json")!=HEAD_HASHES[seed] or result["config"]!=configuration(seed,10000,"coordinate-sl1"):
            raise ValueError("Require the pinned completed Smooth L1 heads")
        if result["prepared_manifest_sha256"]!=sha256(original/"prepared"/str(seed)/"manifest.json"):
            raise ValueError("Prepared data differs from head training")
    if dataset is not None:
        m = json.loads((dataset/"manifest.json").read_text())
        if any(sha256(dataset/"manifest.json")!=x["hashes"]["dataset_manifest"] for x in manifests.values()):
            raise ValueError("Reference mechanism dataset differs from prepared source")
        if sha256(dataset/"mechanism.yaml")!=m["source_manifest"]["mechanism_sha256"]:
            raise ValueError("Reference mechanism changed")
    return manifests


def checked_plan(path):
    result = json.loads((path/"plan.json").read_text())
    verified = json.loads((path/"verification.json").read_text())
    if verified["plan_sha256"]!=sha256(path/"plan.json"):
        raise ValueError("Frozen plan changed")
    for name,digest in result["artifacts"].items():
        if Path(name).name!=name or sha256(path/name)!=digest:
            raise ValueError("Frozen selection artifact changed")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original","candidate","dataset","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new frozen selection directory")
    manifests = inputs(args.original,args.candidate,args.dataset)
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit scientific source first")
    tail_root = args.candidate/"tail"
    tail = json.loads((tail_root/"report.json").read_text())
    if json.loads((tail_root/"verification.json").read_text())["report_sha256"]!=sha256(tail_root/"report.json"):
        raise ValueError("Tail report changed")
    training,tails,unseen = {},{},{}
    for seed in SEEDS:
        name = f"{seed}-tail-components.json"
        if sha256(tail_root/name)!=tail["artifacts"][name]:
            raise ValueError("Recorded tail components changed")
        tails[seed] = json.loads((tail_root/name).read_text())
        data = load_arrays(args.original/"prepared"/str(seed)/"training.npz")
        unseen[seed] = unseen_indices(data["snapshot"])
        training[seed] = {k:v[:10000] for k,v in data.items()}
    species = manifests[SEEDS[0]]["species"]
    records = audit_selection(tails,training,species)
    states,ordinary,bases = [],[],[]
    for record in records:
        rows = training[record["seed"]]
        row = record["training_row"]
        if int(rows["source_indices"][row])!=record["source_row"]:
            raise ValueError("Tail source identity differs")
        states.append(rows["states"][row]); ordinary.append(rows["delta"][row]); bases.append(rows["base"][row])
    print(json.dumps(dict(status="planned",tail_states=64,control_states=16,unseen_rows={s:len(ix) for s,ix in unseen.items()},gammas=GAMMAS)),flush=True)
    if not args.execute:
        return
    args.output.mkdir(parents=True)
    np.savez(args.output/"selections.npz",**{f"unseen_{s}":v for s,v in unseen.items()},
             audit_states=np.asarray(states),audit_ordinary=np.asarray(ordinary),audit_base=np.asarray(bases))
    save(args.output/"audit-selection.json",records)
    dataset_manifest = json.loads((args.dataset/"manifest.json").read_text())
    plan = dict(source=source,species=species,gammas=GAMMAS,head_hashes=HEAD_HASHES,
        prepared_hashes={s:sha256(args.original/"prepared"/str(s)/"manifest.json") for s in SEEDS},
        dataset_manifest_sha256=sha256(args.dataset/"manifest.json"),
        mechanism_sha256=dataset_manifest["source_manifest"]["mechanism_sha256"],
        interval_s=dataset_manifest["config"]["interval_s"],tail_report_sha256=sha256(tail_root/"report.json"),
        artifacts={q.name:sha256(q) for q in args.output.iterdir()})
    save(args.output/"plan.json",plan)
    save(args.output/"verification.json",dict(plan_sha256=sha256(args.output/"plan.json")))


if __name__=="__main__":
    main()
