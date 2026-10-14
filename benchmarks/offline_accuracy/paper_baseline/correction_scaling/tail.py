"""Inspect existing correction tails and a training-mean constant control; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.flame_conditioning.extract import sha256,source_revision
from benchmarks.offline_accuracy.refinement.run import save
from benchmarks.offline_accuracy.paired.runtime import configure
from ..matched_work.run import checked_result
from ..resolved_fit.run import require_idle
from ..adaptive.model import base_preprocessing
from .run import prepared_manifest,load_arrays
from .model import load_base,load_head,base_prediction
from .plan import SEEDS,configuration
from .coverage import magnitude_bins,LABELS
from .tail_metrics import relationships,physical_scores,constant_scores,selected_tail,grouped_tail


def audit_status(rows,audit,active):
    status = np.full(rows["delta"][:,active].shape,"unaudited",dtype="<U24")
    lookup = {int(s):i for i,s in enumerate(rows["source_indices"])}
    for record in audit["records"]:
        if record["split"]!="train" or record["source_row"] not in lookup:
            continue
        row = lookup[record["source_row"]]
        if record["status"]!="checked":
            status[row] = "audit not checked"
            continue
        reference = np.asarray(record["reference_delta"])[active]
        start = rows["states"][row,2:][active]
        uncertainty = np.maximum.reduce([np.asarray(record["uncertainty_estimate"])[active],
            abs(np.spacing(start)),abs(np.spacing(start+reference))])
        status[row] = np.where(uncertainty<=.1*(1e-15+.1*abs(reference)),"audited resolved","audited unresolved")
    return status


def inspect_seed(root,seed,destination):
    prepared = root/"prepared"/str(seed)
    manifest = prepared_manifest(root/"prepared",seed)
    cfg = configuration(seed,10000,"direct")
    active = np.array([s!="AR" for s in manifest["species"]])
    species = np.asarray(manifest["species"])[active]
    training = {k:v[:10000] for k,v in load_arrays(prepared/"training.npz").items()}
    development = load_arrays(prepared/"development.npz")
    b,d,c = training["base"][:,active],training["delta"][:,active],training["target"]
    base,prep = load_base(prepared,seed)
    replay,inverse = base_prediction(base,base_preprocessing(prep),training["states"],seed)
    np.testing.assert_array_equal(replay,training["base"])
    inverse = inverse[:,active]
    status = audit_status(training,json.loads((prepared/"audit.json").read_text()),active)
    relations = relationships(b,d)
    rows,columns = selected_tail(c)
    square = c[rows,columns]**2
    total = float(np.sum(c**2))
    bins = magnitude_bins(d)
    report = dict(seed=seed,prepared_manifest_sha256=sha256(prepared/"manifest.json"),
        training_states=len(d),components=int(d.size),selected_components=len(rows),
        selected_distinct_states=len(np.unique(rows)),top_mse_share=float(square.sum()/total),
        exact_base_replay=True,reference_scope="Existing audit only; unaudited is not incorrect",
        relationships=grouped_tail(relations[rows,columns],rows,square,total),
        species=grouped_tail(species[columns],rows,square,total),
        magnitudes=grouped_tail(np.asarray(LABELS)[bins[rows,columns]],rows,square,total),
        snapshots=grouped_tail(training["snapshot"][rows],rows,square,total),
        inverse_domain=grouped_tail(inverse[rows,columns],rows,square,total),
        reference_audit=grouped_tail(status[rows,columns],rows,square,total))
    records = [dict(training_row=int(r),source_row=int(training["source_indices"][r]),
        species=str(species[j]),reference=float(d[r,j]),base=float(b[r,j]),coordinate_target=float(c[r,j]),
        relationship=str(relations[r,j]),reference_bin=LABELS[bins[r,j]],
        snapshot=str(training["snapshot"][r]),temperature_K=float(training["states"][r,0]),
        pressure_Pa=float(training["states"][r,1]),inverse_domain_corrected=bool(inverse[r,j]),
        reference_audit=str(status[r,j])) for r,j in zip(rows,columns)]
    save(destination/f"{seed}-tail-components.json",records)
    mean = c.mean(axis=0)
    result_dir = root/"10000"/f"{seed}-direct"
    result = checked_result(result_dir)
    if result["prepared_manifest_sha256"]!=report["prepared_manifest_sha256"] or result["config"]!=cfg:
        raise ValueError("Saved MSE head has a different recipe or data")
    model = load_head(result_dir,prep,cfg)
    trained = load_arrays(result_dir/"training-predictions.npz")
    np.testing.assert_array_equal(trained["source_indices"],training["source_indices"])
    fitted_c = trained["coordinate"]
    bias = model[-1].bias.detach().double().cpu().numpy()
    fitted_mean = fitted_c.mean(axis=0)
    report["head_result_sha256"] = sha256(result_dir/"result.json")
    report["species_means"] = [dict(species=str(name),target_mean=float(mean[j]),
        trained_output_mean=float(fitted_mean[j]),final_layer_bias=float(bias[j])) for j,name in enumerate(species)]
    report["output_variation"] = dict(mean_shift_energy=float(np.mean(fitted_mean**2)),
        input_dependent_energy=float(np.mean((fitted_c-fitted_mean)**2)),
        note="Output variation, not a causal attribution of network parameter updates")
    report["constant_control"] = {}
    for split,data in (("training",training),("development",development)):
        truth,old = data["delta"][:,active],data["base"][:,active]
        report["constant_control"][split] = constant_scores(old,truth,data["target"],mean,cfg)
        report["constant_control"][split]["base"] = physical_scores(old,truth,old)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Diagnostic output must be new")
    for seed in SEEDS:
        prepared_manifest(args.root/"prepared",seed)
        checked_result(args.root/"10000"/f"{seed}-direct")
    print(json.dumps(dict(status="planned",training_rows=10000,seeds=SEEDS)),flush=True)
    if not args.execute:
        return
    source = source_revision()
    if source["dirty"]:
        raise ValueError("Commit diagnostic source first")
    require_idle()
    runtime = configure()
    args.output.mkdir(parents=True)
    save(args.output/"environment.json",runtime)
    report = dict(status="complete",source=source,seeds=[])
    for seed in SEEDS:
        result = inspect_seed(args.root,seed,args.output)
        report["seeds"].append(result)
        print(json.dumps(dict(seed=seed,top_mse_share=result["top_mse_share"],
            relationships=result["relationships"],constant_control=result["constant_control"])),flush=True)
    report["artifacts"] = {p.name:sha256(p) for p in args.output.iterdir()}
    save(args.output/"report.json",report)
    save(args.output/"verification.json",dict(status="verified",report_sha256=sha256(args.output/"report.json")))


if __name__=="__main__":
    main()
