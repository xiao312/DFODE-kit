"""Compare isolated study/tight CVODE restarts; never modify either case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.copy_case import chemistry_dictionary
from benchmarks.flame_conditioning.extract import sha256, source_revision
from benchmarks.flame_conditioning.review_cfd import review_case, scalar_field


def check_inputs(study, tight):
    """Permit only explicit chemistry tolerance/path changes, not changed states."""
    plans = [json.loads((case / "preparation.json").read_text()) for case in (study, tight)]
    for key in ("original_sha256", "mechanism_sha256", "steps", "start_time_s", "interval_s"):
        if plans[0][key] != plans[1][key]:
            raise ValueError(f"CFD controls differ in {key}")
    chemistry_file = "constant/CanteraTorchProperties"
    inputs = []
    for case, plan, preset in zip((study, tight), plans, ("study", "tight")):
        expected = chemistry_dictionary(case / "mechanism.yaml", preset)
        if (case / chemistry_file).read_text() != expected:
            raise ValueError("Chemistry dictionary differs from the declared CVODE-only preset")
        hashes = plan["prepared_sha256"]
        for name, digest in hashes.items():
            if sha256(case / name) != digest:
                raise ValueError(f"Prepared CFD input changed: {name}")
        inputs.append({name: digest for name, digest in hashes.items() if name != chemistry_file})
    if inputs[0] != inputs[1]:
        raise ValueError("CFD prepared inputs differ beyond the chemistry dictionary")
    return plans


def difference_metrics(study_fields, tight_fields, species):
    if study_fields.shape != tight_fields.shape or study_fields.shape[1] != 2 + len(species):
        raise ValueError("Comparison field shapes do not match")
    if not np.isfinite(study_fields).all() or not np.isfinite(tight_fields).all():
        raise ValueError("Comparison fields must be finite")
    difference = np.abs(study_fields - tight_fields)
    budget = 1e-12 + 1e-6 * np.abs(tight_fields[:, 2:])
    active = np.asarray([name != "AR" for name in species])
    return {"temperature_max_abs_K": float(difference[:, 0].max()),
            "pressure_max_abs_Pa": float(difference[:, 1].max()),
            "species_max_abs": float(difference[:, 2:].max()),
            "non_argon_species_budget_p99": float(np.quantile((difference[:, 2:] / budget)[:, active], .99)),
            "per_species_max_abs": dict(zip(species, map(float, difference[:, 2:].max(axis=0))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("tight", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    study, tight = args.study.resolve(), args.tight.resolve()
    plans = check_inputs(study, tight)
    reviews = [review_case(case, args.original) for case in (study, tight)]
    gas = ct.Solution(str(study / "mechanism.yaml"))
    final = format(reviews[0]["final_time_s"], ".9g")
    fields = [np.column_stack([scalar_field(case / final / name, 500)
                               for name in ["T", "p"] + gas.species_names]) for case in (study, tight)]
    result = {"status": "complete", "source": source_revision(), "steps": plans[0]["steps"],
              "cells": 500, "final_time_s": reviews[0]["final_time_s"],
              "preparation_sha256": [sha256(case / "preparation.json") for case in (study, tight)],
              "solver_log_sha256": [review["solver_log_sha256"] for review in reviews],
              "original_files_unchanged": True, "original_files_rechecked": len(plans[0]["original_sha256"]),
              "tolerances": {"study": {"rtol": 1e-6, "atol": 1e-10}, "tight": {"rtol": 1e-12, "atol": 1e-21}},
              "final_field_difference": difference_metrics(*fields, gas.species_names),
              "qualification": "Difference of two fixed-step CVODE-only runs, not truth, an error bound, mesh/time convergence, or neural CFD validation"}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
