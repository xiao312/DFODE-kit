"""Repair only the documented heat-diagnostic cancellation failure; dry by default."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.metrics import physical_scores
from benchmarks.offline_accuracy.refinement.run import save
from .run import load_inputs, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("local-state", "local-asinh"), required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path("runs/offline-accuracy")
    args.dataset = root / "20261009/preparation/dataset"
    args.audit = root / "20261009/preparation/audit"
    args.base = root / f"20261009/seed-{args.seed}/training"
    args.refined = root / f"refinement-20261009/seed-{args.seed}/long-state-boxcox"
    args.output = args.campaign / f"seed-{args.seed}/{args.variant}"
    path = args.output / "result.json"
    original = json.loads(path.read_text())
    backup = args.output / "result-before-heat-fix.json"
    if (original["status"] != "failed" or "Saved physical metric differs: heat_error_rms" not in original.get("error", "")
            or backup.exists() or (args.output / "verification.json").exists()):
        raise ValueError("Require the untouched known heat-error failure")
    for name, digest in original["artifacts"].items():
        if sha256(args.output / name) != digest:
            raise ValueError(f"Changed artifact: {name}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    training, validation, physics, audit, model, p, _, hashes = load_inputs(args)
    if original["hashes"] != hashes:
        raise ValueError("Original data identities changed")
    print(json.dumps(dict(action="recheck-heat-only", apply=args.apply, original_sha256=sha256(path))))
    if not args.apply:
        return
    shutil.copyfile(path, backup)
    result = dict(original)
    for split, rows in (("training", training), ("validation", validation)):
        with np.load(args.output / f"{split}-predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(rows["source_indices"], saved["source_indices"])
            result[f"{split}_physical"] = physical_scores(saved["prediction"], rows["delta"], rows["states"], saved["correction"], **physics)
    result.pop("error")
    result.update(status="complete", diagnostic_repair=dict(source=source_revision(),
        original_sha256=sha256(backup), reason="Accumulate species error before heat contraction; predictions and acceptance unchanged"))
    save(path, result)
    try:
        checked = verify(args, training, validation, physics, audit, model, p, hashes)
        save(args.output / "verification.json", checked)
    except Exception as error:
        result.update(status="failed", error=str(error))
        save(path, result)
        raise
    print(json.dumps(dict(status="verified", seed=args.seed, name=args.variant)))


if __name__ == "__main__":
    main()
