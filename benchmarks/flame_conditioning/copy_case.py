"""Prepare an allowlisted copy of an existing 1D restart; never run its scripts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import cantera as ct

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.extract import sha256, source_revision

MECHANISM_SHA256 = "26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0"
MESH_FILES = ("points", "faces", "owner", "neighbour", "boundary")
CONSTANT_FILES = ("g", "thermophysicalProperties", "turbulenceProperties", "combustionProperties")


def header(name, location):
    return f'''FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "{location}";
    object {name};
}}
'''


def control_dictionary(steps):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("Only 1 to 100 steps are allowed in this compatibility runner")
    return header("controlDict", "system") + f'''
application dfLowMachFoam;
startFrom startTime;
startTime 0.0025;
stopAt endTime;
endTime {0.0025 + steps * 1e-6:.8f};
deltaT 1e-6;
writeControl timeStep;
writeInterval {min(steps, 25)};
purgeWrite 0;
writeFormat ascii;
writePrecision 17;
writeCompression off;
timeFormat general;
timePrecision 9;
runTimeModifiable false;
adjustTimeStep off;
functions {{}}
'''


def chemistry_dictionary(mechanism):
    return header("CanteraTorchProperties", "constant") + f'''
chemistry on;
CanteraMechanismFile "{mechanism}";
transportModel "Mix";
odeCoeffs
{{
    relTol 1e-6;
    absTol 1e-10;
}}
inertSpecie AR;
splittingStrategy off;
TorchSettings
{{
    torch false;
    GPU false;
    log false;
    torchModel "unused";
    frozenTemperature 0;
    inferenceDeltaTime 1e-6;
    coresPerNode 1;
}}
loadbalancing
{{
    active false;
    log false;
    algorithm allAverage;
}}
'''


def adapt_energy_solvers(text):
    old = '"(U|ha|k|epsilon)'
    count = text.count(old)
    if count != 2:
        raise ValueError("Expected two original U/ha solver patterns; inspect the source instead of guessing")
    return text.replace(old, '"(U|h|hs|ha|k|epsilon)')


def inspect_source(source, species):
    files = [Path("0.0025") / name for name in ["T", "p", "U"] + list(species)]
    files += [Path("constant/polyMesh") / name for name in MESH_FILES]
    files += [Path("constant") / name for name in CONSTANT_FILES]
    files += [Path("system") / name for name in ("fvSchemes", "fvSolution")]
    for relative in files:
        path = source / relative
        if not path.is_file():
            raise ValueError(f"Required source file missing: {relative}")
        text = path.read_text()
        if re.search(r"#(?:include|codeStream|calc)|\bcoded\w*", text):
            raise ValueError(f"Unsupported executable/include directive in {relative}")
        if relative.parts[0] == "0.0025":
            expected = "volVectorField" if relative.name == "U" else "volScalarField"
            if not re.search(rf"\bclass\s+{expected}\s*;", text):
                raise ValueError(f"Unexpected field class in {relative}; refuse unsafe field-name collision")
    adapt_energy_solvers((source / "system/fvSolution").read_text())
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-case", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    source, output = args.source_case.resolve(), args.output.resolve()
    if output.exists() or source == output or source in output.parents:
        parser.error("Output must be new and outside the original case")
    control = control_dictionary(args.steps)
    if sha256(args.mechanism) != MECHANISM_SHA256:
        raise ValueError("This compatibility case requires the inspected study mechanism hash")
    gas = ct.Solution(str(args.mechanism))
    files = inspect_source(source, gas.species_names)
    plan = {"source_case": str(source), "output_case": str(output), "source": source_revision(),
            "steps": args.steps, "start_time_s": .0025, "interval_s": 1e-6,
            "mechanism_sha256": MECHANISM_SHA256, "copied_files": [str(path) for path in files],
            "original_sha256": {str(path): sha256(source / path) for path in files},
            "changes": ["h/hs energy solver patterns", "literal bounded control dictionary",
                        "ANN/GPU/load balancing/functions disabled", "17-digit new output"],
            "qualification": "Compatibility restart; not exact paper reproduction or learned CFD validation"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.apply:
        return
    output.mkdir(parents=True, exist_ok=False)
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    shutil.copyfile(args.mechanism, output / "mechanism.yaml")
    solution = output / "system/fvSolution"
    solution.write_text(adapt_energy_solvers(solution.read_text()))
    (output / "system/controlDict").write_text(control)
    (output / "constant/CanteraTorchProperties").write_text(chemistry_dictionary(output / "mechanism.yaml"))
    plan["prepared_sha256"] = {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}
    (output / "preparation.json").write_text(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
