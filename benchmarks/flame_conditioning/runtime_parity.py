"""Run with the unchanged CFD runtime's Python 3.8 and Cantera 2.6."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import cantera as ct
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest["status"] not in ("complete", "complete_with_exclusions"):
        raise ValueError("Dataset is incomplete")
    for name in ("inputs", "labels"):
        if digest(args.dataset / "validation" / (name + ".npz")) != manifest["splits"]["validation"][name + "_sha256"]:
            raise ValueError("Validation artifact hash mismatch")
    mechanism = args.dataset / "mechanism.yaml"
    if digest(mechanism) != manifest["source_manifest"]["mechanism_sha256"]:
        raise ValueError("Mechanism hash mismatch")
    inputs = np.load(args.dataset / "validation/inputs.npz", allow_pickle=False)
    labels = np.load(args.dataset / "validation/labels.npz", allow_pickle=False)
    ids = np.flatnonzero(labels["accepted"])
    order = ids[np.argsort(inputs["states"][ids, 0], kind="stable")]
    selected = order[np.linspace(0, len(order) - 1, min(32, len(order))).astype(int)]
    plan = {"runtime_cantera": ct.__version__, "python": platform.python_version(),
            "research_cantera": manifest["cantera"], "dataset_sha256": digest(args.dataset / "manifest.json"),
            "mechanism_sha256": digest(mechanism), "selected_validation_rows": selected.tolist(),
            "rtol": 1e-12, "atol": 1e-21, "interval_s": manifest["config"]["interval_s"]}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    if ct.__version__ != "2.6.0":
        raise ValueError("This parity check requires the unchanged installed Cantera 2.6.0")
    started = time.monotonic()
    result = {"status": "running", "plan": plan, "records": [], "failures": []}
    for index in selected:
        row = inputs["states"][index]
        record = {"row": int(index), "temperature_K": float(row[0])}
        try:
            gas = ct.Solution(str(mechanism))
            if gas.species_names != manifest["source_manifest"]["species_names"]:
                raise ValueError("Runtime species order differs")
            gas.set_unnormalized_mass_fractions(row[2:])
            gas.TP = row[0], row[1]
            initial_density = gas.density
            reactor = ct.Reactor(gas, energy="off")
            network = ct.ReactorNet([reactor])
            network.rtol, network.atol, network.max_steps = 1e-12, 1e-21, 100000
            network.advance(plan["interval_s"])
            delta = reactor.thermo.Y - row[2:]
            difference = np.abs(delta - labels["delta"][index])
            budget = 1e-12 + 1e-6 * np.abs(row[2:])
            record.update(delta=delta.tolist(), difference_budget=(difference / budget).tolist(),
                          difference_budget_max=float(np.max(difference / budget)),
                          mass_drift=float(abs(delta.sum())), minimum_endpoint=float(reactor.thermo.Y.min()),
                          temperature_change_K=float(reactor.T - row[0]),
                          relative_density_change=float(reactor.density / initial_density - 1))
        except Exception as error:
            record["error"] = str(error)
            result["failures"].append({"row": int(index), "error": str(error)})
        result["records"].append(record)
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    result["status"] = "incomplete" if result["failures"] else "complete"
    if not result["failures"]:
        values = np.array([record["difference_budget"] for record in result["records"]])
        result.update(max_difference_budget=float(values.max()), budget_fit_fraction=float(np.mean(values <= .01)),
                      scope="Empirical version/interface agreement on a selected subset, not every label")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "plan")}), flush=True)


if __name__ == "__main__":
    main()
