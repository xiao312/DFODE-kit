"""Compare 1/4/8 reference workers on identical existing rows; never train."""
import argparse
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from benchmarks.flame_conditioning.data import load_dataset
from benchmarks.flame_conditioning.extract import sha256, source_revision
from .storage import contract, save_json
from .worker import chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 64 <= args.rows <= 8192:
        parser.error("Choose a new output and 64 to 8192 rows")
    _, _, manifest = load_dataset(args.dataset)
    with np.load(args.dataset / "train/inputs.npz", allow_pickle=False) as inputs:
        indices = np.linspace(0, len(inputs["states"])-1, args.rows, dtype=int)
        if len(np.unique(indices)) != args.rows:
            parser.error("Need enough distinct saved rows")
        states = inputs["states"][indices]
    with np.load(args.dataset / "train/labels.npz", allow_pickle=False) as labels:
        reference, mask = labels["delta"][indices], labels["accepted"][indices]
    print(json.dumps(dict(rows=args.rows, workers=[1, 4, 8], repetitions=2, execute=args.execute)))
    if not args.execute:
        return
    result = dict(status="running", source=source_revision(),
                  dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
                  contract=contract(manifest["config"], manifest["source_manifest"]),
                  source_rows=indices.tolist(), measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    try:
        for repetition, order in enumerate(((1, 4, 8), (8, 4, 1))):
            for workers in order:
                predicted = np.full_like(reference, np.nan)
                accepted = np.zeros_like(mask)
                ranges = [(i, min(i+128, len(states))) for i in range(0, len(states), 128)]
                started = time.monotonic()
                for start, delta, flags, _ in chunks(args.dataset / "mechanism.yaml", manifest["config"],
                                                    states, ranges, workers, started+300):
                    predicted[start:start+len(delta)] = delta
                    accepted[start:start+len(delta)] = flags
                elapsed = time.monotonic()-started
                np.testing.assert_array_equal(predicted, reference)
                np.testing.assert_array_equal(accepted, mask)
                result["measurements"].append(dict(workers=workers, repetition=repetition,
                    rows=len(states), seconds=elapsed, rows_per_second=len(states)/elapsed,
                    exact_serial_labels=True, exact_acceptance_flags=True))
                save_json(args.output, result)
                print(json.dumps(result["measurements"][-1]), flush=True)
        result["status"] = "verified"
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        save_json(args.output, result)


if __name__ == "__main__":
    main()
