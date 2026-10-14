"""Non-learned rate-based controls. No acceptance claim without saved verification."""
import argparse
import json
import time
from pathlib import Path

import cantera as ct
import numpy as np

from benchmarks.flame_conditioning.chemistry import set_state
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores
from benchmarks.offline_accuracy.metrics import summarize
from benchmarks.offline_accuracy.verify_evaluation import check_summary


def frozen_step(fractions, net_rate, destruction_rate, interval, mode):
    if mode == "rate-euler":
        return interval*net_rate
    if mode != "frozen-exponential":
        raise ValueError("Unknown physics-prior mode")
    # dy/dt = production - k*y, with frozen production/k. Use the net rate
    # directly, avoiding another cancellation in production - k*y.
    coefficient = np.divide(destruction_rate, fractions,
                            out=np.zeros_like(fractions), where=fractions > 0)
    if np.any((fractions == 0) & (destruction_rate != 0)):
        raise ValueError("A zero species has nonzero destruction; frozen-rate model is undefined")
    z = interval*coefficient
    factor = np.divide(-np.expm1(-z), z, out=np.ones_like(z), where=z != 0)
    return interval*net_rate*factor


class Predictor:
    def __init__(self, mechanism, interval, mode):
        self.gas = ct.Solution(str(mechanism))
        self.interval, self.mode = interval, mode

    def __call__(self, states):
        values = []
        for row in states:
            set_state(self.gas, dict(T=row[0], P=row[1], Y=row[2:]))
            conversion = self.gas.molecular_weights/self.gas.density
            value = frozen_step(row[2:], self.gas.net_production_rates*conversion,
                self.gas.destruction_rates*conversion, self.interval, self.mode)
            values.append(value)
        result = np.array(values)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite physics-prior prediction")
        return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def main():
    from benchmarks.offline_accuracy.evaluate import audit_subset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, manifest = load_dataset(args.dataset)
    audit = json.loads((args.audit / "summary.json").read_text())
    if (audit.get("status") != "complete" or not audit.get("reference_subset_pass")
            or audit.get("dataset_manifest_sha256") != sha256(args.dataset / "manifest.json")):
        raise ValueError("Require matching complete data and passing reference audit")
    modes = ("rate-euler", "frozen-exponential")
    print(json.dumps(dict(modes=modes, states=len(data["validation"]["states"]), dry_run=args.dry_run)))
    if args.dry_run:
        return
    args.output.mkdir(parents=True)
    result = dict(status="running", source=source_revision(), cantera=ct.__version__,
        dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
        audit_sha256=sha256(args.audit / "summary.json"),
        scope="Non-learned mechanism-dependent controls, not certified integration or network gains", models=[])
    save(args.output / "summary.json", result)
    try:
        for mode in modes:
            predictor = Predictor(args.dataset / "mechanism.yaml", physics["interval"], mode)
            record = dict(name=mode)
            for split in ("train", "validation"):
                rows = data[split]
                predicted = predictor(rows["states"])
                np.savez_compressed(args.output / f"{mode}-{split}.npz", prediction=predicted, source_indices=rows["source_indices"])
                record[split] = summarize(predicted, rows["delta"], physics["species_names"])
                record[f"{split}_physical"] = physical_scores(predicted, rows["delta"], rows["states"], np.zeros_like(predicted, dtype=bool), **physics)
                check_summary(predicted, rows["delta"], record[split], physics["species_names"])
                assert_scores(recompute(rows["states"], predicted, rows["delta"], args.dataset / "mechanism.yaml", physics["interval"]), record[f"{split}_physical"])
                np.testing.assert_array_equal(predicted, Predictor(args.dataset / "mechanism.yaml", physics["interval"], mode)(rows["states"]))
            selected, reference, uncertainty = audit_subset(rows, audit)
            record["audited_subset"] = summarize(predicted[selected], reference, physics["species_names"], uncertainty)
            check_summary(predicted[selected], reference, record["audited_subset"], physics["species_names"], uncertainty)
            timings = []
            for _ in range(5):
                start = time.process_time()
                predictor(rows["states"])
                timings.append(time.process_time()-start)
            record["inference_cpu_ms_per_state"] = 1000*float(np.median(timings))/len(rows["states"])
            result["models"].append(record)
        result["artifacts"] = {p.name: sha256(p) for p in args.output.glob("*.npz")}
        result["status"] = "complete"
        save(args.output / "summary.json", result)
        save(args.output / "verification.json", dict(status="verified", summary_sha256=sha256(args.output / "summary.json"),
            independent_counts=True, physical_checks=True, prediction_replay="exact"))
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(args.output / "summary.json", result)
        raise
    print(json.dumps(dict(status="verified", models=len(result["models"]))))


if __name__ == "__main__":
    main()
