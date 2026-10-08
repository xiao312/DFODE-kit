"""Check preselected historical weights on the audited 1D validation domain."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.evaluate_heldout import hybrid_prediction
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.heldout import freeze_historical
from benchmarks.flame_conditioning.historical import load_historical, MODES
from benchmarks.flame_conditioning.metrics import physical_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--historical", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    data, physics, _ = load_dataset(args.dataset)
    audit = json.loads((args.audit / "summary.json").read_text())
    if (audit.get("status") != "complete" or audit.get("reference_subset_pass") is not True
            or audit["dataset_manifest_sha256"] != sha256(args.dataset / "manifest.json")):
        raise ValueError("Require the passing independent audit for this exact dataset")
    plan = freeze_historical(args.historical)
    print(json.dumps({"historical": plan, "modes": MODES}), flush=True)
    if args.dry_run:
        return
    torch.set_num_threads(1)
    args.output.mkdir(parents=True)
    result = {"source": source_revision(), "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
              "historical": plan, "models": [], "scope": "Historical training overlap not excluded; not independent validation"}
    states, reference = data["validation"]["states"], data["validation"]["delta"]
    predictors = {}

    def score(name, prediction, correction, diagnostics):
        result["models"].append({"name": name, "validation": physical_scores(prediction, reference, states, correction, **physics),
                                  "diagnostics": diagnostics})
        np.savez_compressed(args.output / f"{name}.npz", prediction=prediction, correction=correction,
                            source_indices=data["validation"]["source_indices"])
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({"name": name, "budget_p99": result["models"][-1]["validation"]["budget_error"]["p99"]}), flush=True)

    for control in plan:
        for mode in MODES:
            predict, _ = load_historical(control["directory"], physics["species_names"], mode)
            predictors[(control["kind"], mode)] = predict
            prediction, correction = predict(states)
            score(f'{control["kind"]}--{mode}', prediction, correction, dict(predict.diagnostics))
    for mode in MODES:
        boxcox, power = predictors.get(("state-boxcox", mode)), predictors.get(("signed-power", mode))
        if boxcox is not None and power is not None:
            score(f"fixed-hybrid--{mode}", *hybrid_prediction(states, boxcox, power),
                  diagnostics={"thresholds_K": [305, 1000], "historical_training_overlap_not_excluded": True})
    result["status"] = "complete"
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
