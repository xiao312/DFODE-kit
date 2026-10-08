"""Read copied CFD evidence and confirm original allowlisted files are unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.extract import sha256, source_revision


def scalar_field(path, count):
    text = Path(path).read_text()
    if not re.search(r"\bformat\s+ascii\s*;", text) or not re.search(r"\bclass\s+volScalarField\s*;", text):
        raise ValueError("Only inspected ASCII scalar fields are supported")
    uniform = re.search(r"\binternalField\s+uniform\s+([^;]+);", text)
    if uniform:
        return np.full(count, float(uniform[1]))
    match = re.search(r"\binternalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if match is None or int(match[1]) != count:
        raise ValueError(f"Unexpected internal field shape: {Path(path).name}")
    values = np.fromstring(match[2], sep=" ")
    if len(values) != count:
        raise ValueError("Scalar field count differs from its declaration")
    return values


def x_cell_centres(case, original, count):
    """Validate the old coordinate-vector C against this nonuniform 1D mesh.

    Read C only as geometry. Never copy it to a species-field destination.
    """
    centre_file = Path(original) / "0/C"
    text = centre_file.read_text()
    if not re.search(r"\bclass\s+volVectorField\s*;", text):
        raise ValueError("The inspected geometry C must be a vector field, not carbon")
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\n\)\s*;", text, re.S)
    if match is None or int(match[1]) != count:
        raise ValueError("Coordinate field shape mismatch")
    centres = np.fromstring(re.sub(r"[()]", " ", match[2]), sep=" ").reshape(count, 3)
    points = (Path(case) / "constant/polyMesh/points").read_text()
    vectors = re.findall(r"\(\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s*\)", points)
    points = np.asarray(vectors, dtype=float)
    interfaces = np.unique(points[:, 0])
    if len(interfaces) != count + 1 or len(points) != 4 * (count + 1):
        raise ValueError("Only the inspected single-row hexahedral mesh is supported")
    expected = .5 * (interfaces[:-1] + interfaces[1:])
    np.testing.assert_allclose(centres[:, 0], expected, rtol=5e-6, atol=1e-9)
    if not np.allclose(centres[:, 1:], centres[0, 1:]):
        raise ValueError("Expected a one-dimensional row of cell centres")
    return expected, centre_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Review output must be new")
    prep = json.loads((args.case / "preparation.json").read_text())
    changed = [name for name, digest in prep["original_sha256"].items() if sha256(args.original / name) != digest]
    if changed:
        raise ValueError(f"Original source hash changed: {changed}")
    if sha256(args.case / "mechanism.yaml") != prep["mechanism_sha256"]:
        raise ValueError("Copied mechanism hash changed")
    log = (args.case / "log.solver").read_text()
    mesh = (args.case / "log.checkMesh").read_text()
    times = re.findall(r"^Time = ([0-9.eE+-]+)$", log, re.M)
    if not times or len(times) != prep["steps"] or not re.search(r"^End\s*$", log, re.M) or "FOAM FATAL" in log:
        raise ValueError("Solver did not complete the planned number of steps")
    final = float(times[-1])
    if abs(final - (prep["start_time_s"] + prep["steps"] * prep["interval_s"])) > 1e-12:
        raise ValueError("Unexpected final time")
    if "Mesh OK" not in mesh:
        raise ValueError("Mesh check did not pass")
    gas = ct.Solution(str(args.case / "mechanism.yaml"))
    centres, centre_file = x_cell_centres(args.case, args.original, 500)
    result = {"source": source_revision(), "status": "complete", "steps": len(times),
              "start_time_s": prep["start_time_s"], "final_time_s": final, "cells": 500,
              "mesh_check": "passed", "original_files_rechecked": len(prep["original_sha256"]),
              "original_files_unchanged": True, "solver_log_sha256": sha256(args.case / "log.solver"),
              "mechanism_sha256": prep["mechanism_sha256"], "states": [], "profiles": [],
              "geometry_coordinate_sha256": sha256(centre_file),
              "scope": "CVODE-only restart compatibility, not coupled neural-model validation"}
    for time_name in ("0.0025", times[-1]):
        fields = np.column_stack([scalar_field(args.case / time_name / name, 500) for name in ["T", "p"] + gas.species_names])
        if not np.isfinite(fields).all() or np.any(fields[:, :2] <= 0):
            raise ValueError("Nonfinite or nonpositive T/p in copied run")
        fractions = fields[:, 2:]
        result["states"].append({"time_s": float(time_name), "temperature_min_K": float(fields[:, 0].min()),
                                 "temperature_max_K": float(fields[:, 0].max()), "pressure_min_Pa": float(fields[:, 1].min()),
                                 "pressure_max_Pa": float(fields[:, 1].max()), "species_min": float(fractions.min()),
                                 "negative_species_components": int(np.sum(fractions < 0)),
                                 "mass_closure_max": float(np.max(np.abs(fractions.sum(axis=1) - 1)))})
        result["profiles"].extend({"cell": index, "position_mm": float(position * 1000),
                                    "time_ms": float(time_name) * 1000, "temperature_K": float(fields[index, 0])}
                                   for index, position in enumerate(centres))
    if not args.dry_run:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key != "profiles"}, indent=2))


if __name__ == "__main__":
    main()
