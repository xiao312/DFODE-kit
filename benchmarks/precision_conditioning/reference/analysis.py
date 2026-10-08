"""Metrics use species components as their denominator; failures are separate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def assess(record, config):
    required = ("absolute18", "absolute21", "step_limited", "radau_check", "radau_reference")
    if any(name not in record["solutions"] for name in required):
        return None
    solutions = record["solutions"]
    reference = np.asarray(solutions["radau_reference"]["delta"])
    uncertainty = np.max([np.abs(np.asarray(solutions[name]["delta"]) - reference)
                          for name in required if name != "radau_reference"], axis=0)
    initial = np.r_[record["state"]["T"], record["state"]["Y"]]
    species = config["species_budget"]
    thermal = config["temperature_budget"]
    weights = np.r_[thermal["atol"] + thermal["rtol"] * abs(initial[0]),
                    species["atol"] + species["rtol"] * abs(initial[1:])]
    spacing = np.maximum(np.abs(np.spacing(initial)), np.abs(np.spacing(initial + reference)))
    budget_fit = uncertainty <= 0.01 * weights
    relative_fit = budget_fit & (np.abs(reference) > 100 * np.maximum(uncertainty, spacing))
    endpoint = np.asarray(solutions["absolute21"]["delta"])
    return {"reference": reference, "uncertainty": uncertainty, "weights": weights,
            "budget_fit": budget_fit, "relative_fit": relative_fit,
            "endpoint_zero_direct_nonzero": (endpoint == 0) & (reference != 0),
            "spacing": spacing}


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "p50": None, "p99": None, "max": None}
    return {"count": int(values.size), "p50": float(np.median(values)),
            "p99": float(np.quantile(values, 0.99)), "max": float(values.max())}


def summarize(records, config, mechanism_info):
    component_rows, solver_rows, failure_rows = [], [], []
    metadata = {item["id"]: item for item in mechanism_info}
    for record in records:
        failure_rows.extend({"record_id": record["id"], "solver": key, "error": message}
                            for key, message in record["failures"].items())
        assessment = assess(record, config)
        for name, solution in record["solutions"].items():
            values = {"record_id": record["id"], "mechanism": record["mechanism"],
                      "solver": name, "seconds": solution["seconds"],
                      **solution["conservation"]}
            if assessment is not None:
                error = np.abs(np.asarray(solution["delta"])[1:] - assessment["reference"][1:])
                scaled = error / assessment["weights"][1:]
                values.update({"species_budget_error_max": float(scaled.max()),
                               "species_budget_error_rms": float(np.sqrt(np.mean(scaled**2))),
                               "species_budget_exceedance_fraction": float(np.mean(scaled > 1)),
                               "temperature_budget_error": float(abs(solution["delta"][0] - assessment["reference"][0]) / assessment["weights"][0])})
            solver_rows.append(values)
        if assessment is None:
            continue
        for index, species in enumerate(metadata[record["mechanism"]]["species"], 1):
            delta = assessment["reference"][index]
            component_rows.append({
                "record_id": record["id"], "mechanism": record["mechanism"], "species": species,
                "interval_s": record["interval_s"], "delta": float(delta),
                "uncertainty_budget": float(assessment["uncertainty"][index] / assessment["weights"][index]),
                "budget_fit": bool(assessment["budget_fit"][index]),
                "relative_fit": bool(assessment["relative_fit"][index]),
                "endpoint_zero_direct_nonzero": bool(assessment["endpoint_zero_direct_nonzero"][index]),
                "reference_zero_estimate": bool(delta == 0),
            })
    count = len(component_rows)
    grouped = []
    for mechanism in metadata:
        for species in metadata[mechanism]["species"]:
            rows = [row for row in component_rows if row["mechanism"] == mechanism and row["species"] == species]
            if rows:
                grouped.append({"mechanism": mechanism, "species": species, "count": len(rows),
                                "budget_fit_fraction": float(np.mean([row["budget_fit"] for row in rows])),
                                "relative_fit_fraction": float(np.mean([row["relative_fit"] for row in rows])),
                                "endpoint_zero_direct_nonzero_count": sum(row["endpoint_zero_direct_nonzero"] for row in rows),
                                "uncertainty_budget_max": max(row["uncertainty_budget"] for row in rows)})
    bins = []
    for lower, upper in [(-np.inf, -32), (-32, -24), (-24, -16), (-16, -8), (-8, np.inf)]:
        rows = [row for row in component_rows if row["delta"] != 0 and lower <= np.log10(abs(row["delta"])) < upper]
        bins.append({"bin": f"[{lower},{upper})", "count": len(rows),
                     "budget_fit_count": sum(row["budget_fit"] for row in rows),
                     "relative_fit_count": sum(row["relative_fit"] for row in rows),
                     "endpoint_zero_direct_nonzero_count": sum(row["endpoint_zero_direct_nonzero"] for row in rows)})
    by_solver = {}
    for solver in sorted({row["solver"] for row in solver_rows}):
        rows = [row for row in solver_rows if row["solver"] == solver]
        by_solver[solver] = {
            "completed": len(rows), "total_seconds": sum(row["seconds"] for row in rows),
            "species_budget_error_max_per_interval": distribution([row["species_budget_error_max"] for row in rows if "species_budget_error_max" in row]),
            "temperature_budget_error": distribution([row["temperature_budget_error"] for row in rows if "temperature_budget_error" in row]),
            "mass_delta_sum_abs_max": max(abs(row["mass_delta_sum"]) for row in rows),
            "element_delta_max": max(row["element_delta_max"] for row in rows),
            "enthalpy_relative_drift_max": max(row["enthalpy_relative_drift"] for row in rows),
            "minimum_mass_fraction": min(row["minimum_mass_fraction"] for row in rows),
        }
    return {"interval_count": len(records), "assessed_species_components": count,
            "assessed_interval_count": len({row["record_id"] for row in component_rows}),
            "budget_fit_count": sum(row["budget_fit"] for row in component_rows),
            "relative_fit_count": sum(row["relative_fit"] for row in component_rows),
            "reference_zero_estimate_count": sum(row["reference_zero_estimate"] for row in component_rows),
            "endpoint_zero_direct_nonzero_count": sum(row["endpoint_zero_direct_nonzero"] for row in component_rows),
            "uncertainty_budget": distribution([row["uncertainty_budget"] for row in component_rows]),
            "solvers": by_solver, "failures": failure_rows, "magnitude_bins": bins,
            "species": grouped}, component_rows, solver_rows


def write_analysis(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    records = [json.loads(line) for line in (output / "intervals.jsonl").read_text().splitlines() if line.strip()]
    summary, components, solvers = summarize(records, manifest["config"], manifest["mechanisms"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, rows in (("components.csv", components), ("solver-metrics.csv", solvers)):
        if not rows:
            continue
        columns = sorted(set().union(*(row.keys() for row in rows)))
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(write_analysis(args.output), allow_nan=False))
