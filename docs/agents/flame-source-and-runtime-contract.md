# Flame source and runtime contract

Checked on 2026-10-08. This note records read-only source checks for the
flame-based research route. It does not claim a reproduction of the Fuel paper.
No original case, dataset, installation, or mechanism was changed.

## Decision

Use the existing NH3/CH4 study assets as the first source of flame states.
Do not substitute a new homogeneous-reactor grid for these states. Label the
states with the chemistry operation that the CFD solver actually uses:
**fixed volume, fixed temperature, and a 1e-6 s chemistry interval**.
The prior constant-pressure, adiabatic pilot tests a different operation.

The paper identifies the Okafor 2018 mechanism and uses flame-derived states,
augmentation, and subsequent reacting-flow validation. It does not establish
that a generic hot-reactor benchmark is an equivalent application test.
[Fuel manuscript](https://arxiv.org/html/2507.08277v2)

## Asset roots and provenance

Machine-specific roots belong in ignored run manifests, not this published
note. The main agent has the absolute paths from the inspection messages.
Use these portable root names:

- `STUDY_ROOT`: the existing `nh3ch4_dnn_project` directory.
- `WORKSPACE`: `${STUDY_ROOT}/60nh3_workspace`.
- `ONE_D_CASE`: `${WORKSPACE}/active_work/1d_flames/1d_flame_60nh3.ULFS`.
- `TWO_D_CASE`: `${WORKSPACE}/active_work/training/60NH3_ER1_URMS1.cvode`.

These are user-owned research assets. Their directory names alone do not prove
that a particular file produced a published figure. Record this qualification
in any result report.

### Mechanism

The following copies have identical SHA-256 hashes:

1. `${WORKSPACE}/scripts.orig/1d_flame_60nh3/Okafor2018_s59r356.yaml`.
2. `${WORKSPACE}/active_work/training/Okafor2018_s59r356.yaml`.
3. `${STUDY_ROOT}/hit_cases_orig/60NH3_ER1_URMS1/60NH3_ER1_URMS1_CVODE/Okafor2018_s59r356.yaml`.

Hash: `26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0`.

An unchanged local copy is in ignored
`runs/sources/nh3ch4-study/Okafor2018_s59r356.yaml`. Loading this copy with
Cantera 3.2.0 gives 59 species and 356 reactions. All 59 species have transport
data. The phase declares ideal-gas thermodynamics and mixture-averaged
transport. The file header records `ck2yaml`, Cantera 2.6.0, input files
`chem.inp`, `therm.dat`, and `trans.dat`, and conversion on 2024-01-10.
The description identifies GRI carbon chemistry and Tian nitrogen chemistry.
These checks establish the exact local artifact; they do not establish
equivalence to an independently obtained publisher supplement.

The original mechanism paper is
[Okafor et al., Combustion and Flame 187, 185–198](https://doi.org/10.1016/j.combustflame.2017.09.002).
The publisher page returned HTTP 403 during this check. The scoped local
search did not locate the three original CHEMKIN input files or a separate
mechanism license. Do not redistribute the YAML in public Git, Pages, or an
image until redistribution rights are established. Internal reuse of the
user-provided asset must retain its hash and source path.

### Original labeling code

Two inspected study scripts use `ct.Reactor(gas, energy='off')`, a 1e-6 s
interval, `rtol=1e-6`, and `atol=1e-10`:

- `${WORKSPACE}/active_work/scripts/cantera_integration.py`.
- `${WORKSPACE}/active_work/dataset/oneStepCanteraCV.py`.

The first script has SHA-256
`600e8f18c2f4a82d0592af6a16b836f5e037d455393d6a8093cd32b5bfb6a1de`.
It stores before/after `[T, p, Y, species_specific_enthalpies]`, giving 240
columns for 59 species. These are direct code observations, not inferred from
the paper. Some old scripts request up to 100 worker processes and have weak
worker-error handling. Do not run them unchanged on the shared server.

## Flame-state sources

### 1D training and validation candidates

`${ONE_D_CASE}/system/blockMeshDict` specifies 500 cells over 44 mm.
`system/sample_config` selects stage 1: 1e-6 s steps, 2500 steps, sampling
each step. Paired sample files are available under:

```text
${ONE_D_CASE}/postProcessing/lineSampleA/<time>/*.xy
${ONE_D_CASE}/postProcessing/lineSampleB/<time>/*.xy
```

The inspected surviving times number 2401, from 0.0001 through 0.0025 s.
The first 99 expected sample times are not present in this directory.
Each inspected pair has 500 rows and matching coordinates. Column 0 is
distance. After removing that coordinate, A contains `T`, `p`, and species
from H2 through HCCOH; B contains N through AR. Concatenation gives
`[T, p] + mechanism.species_names`, or 61 columns.

Keep source time, coordinate, and row ID. Split source snapshots before
interpolation or perturbation. Nearby time samples remain correlated; call
this a within-case test, not independent flame generalization.

The legacy aggregator
`${WORKSPACE}/active_work/scripts/sample_n_interpolation_aug_1d.py` has hash
`099a066330d571a9451ff13c110b5dc12b183cb086b635c24a56535a685f0e44`.
It visits time directories without numerical sorting, sorts each snapshot by
temperature, and concatenates states without preserving time IDs. Therefore,
do not infer physical time from row number in its combined NPY output.
Reading selected paired XY files avoids this ambiguity.

The old raw NPY is
`${WORKSPACE}/active_work/dataset/1Dflame_1d_flame_60nh3.ULFS_0.0-2.5ms_raw.npy`:
shape `(1250000, 61)`, dtype FP64, 610,000,128 bytes. Existing labeled arrays
reach 15–20 GB. There is no need to copy or load them for the bounded pilot.

### Input precision caveat

Three inspected snapshots, at 0.0001, 0.0013, and 0.0025 s, have no negative
mass fractions. Their maximum mass-fraction closure errors are respectively
`9.8454e-7`, `1.0352e-6`, and `9.5111e-7`. Sample text contains approximately
six significant digits. FP64 storage does not restore lost source digits.

For a new pilot, preserve raw states, record explicit normalization, and
relabel the normalized states. Separate reference-integration accuracy from
uncertainty in the original CFD state. Do not clip negative values silently.
These three snapshots span approximately 300–2169 K. Their fresh-side NH3 and
CH4 mass fractions are consistent with a 60/40 molar fuel blend; this is a
check on the recovered case, not proof about every historical run.

### 2D held-out candidates

`${TWO_D_CASE}` contains reconstructed fields at 0.001, 0.002, 0.003, and
0.004 s. The existing source array
`${WORKSPACE}/active_work/dataset/60NH3_ER1_URMS1.cvode_0.002.npy` has shape
`(262144, 61)`, dtype FP64, and size 127,926,400 bytes.

The first three cells of every NPY column match the corresponding named
OpenFOAM field exactly in `[T, p] + mechanism.species_names` order.
This was a schema check only: no learned model was scored and no accuracy
metrics were computed on the 2D array. Keep it sealed for subsequent testing.
The similarly named `hit_cases_orig` case contains initial fields only; use
the active-work case when locating evolved snapshots.

## Runtime boundary

Public upstream source was inspected at commit
`7cb3294b275525cb8829ada3d3fbc067ab9323eb`. The main agent separately checks
the installed image, because the two versions need not be identical.

The chemistry model creates a Cantera `Reactor` with energy disabled. For
each cell it initializes T, p, and Y, advances chemistry, then returns
`RR_i = rho_initial * (Y_after_i - Y_before_i) / dt`. Heat release is formed
from formation enthalpies as `-sum(hc_i * RR_i)`.
[Chemistry implementation](https://github.com/deepmodeling/deepflame-dev/blob/7cb3294b275525cb8829ada3d3fbc067ab9323eb/src/dfChemistryModel/dfChemistryModel.C)

The Python bridge packs `[T, p/101325, Y..., rho]`; its pressure field is in
atmospheres, unlike stored training-state pressure in Pa. Species follow the
mechanism order. The bridge consumes species mass source terms, not a learned
temperature endpoint.
[Packing code](https://github.com/deepmodeling/deepflame-dev/blob/7cb3294b275525cb8829ada3d3fbc067ab9323eb/src/dfChemistryModel/torchFunctions.H),
[Python interface](https://github.com/deepmodeling/deepflame-dev/blob/7cb3294b275525cb8829ada3d3fbc067ab9323eb/src/dfChemistryModel/pytorchFunctions.H)

The energy equation handles sensible and absolute enthalpy differently:
`hs` uses chemical heat release, while `ha` does not add that term again.
An adapter must not independently add a predicted temperature change or
double-count formation enthalpy.
[Energy equation](https://github.com/deepmodeling/deepflame-dev/blob/7cb3294b275525cb8829ada3d3fbc067ab9323eb/applications/solvers/dfLowMachFoam/EEqn.H)

The old study template
`${WORKSPACE}/scripts.orig/templates/60NH3_ER1_URMS1/inference.py` takes the
absolute value of all inputs, overwrites pressure with 101325 Pa, predicts
58 non-argon species, renormalizes their endpoint fractions, and returns
`rho_initial * delta_Y / dt`. It is useful historical evidence, not a safe
generic adapter. Implement explicit units, normalization, species mapping,
and invalid-state policy instead of copying these assumptions silently.

## Next checks

1. Pin the local mechanism hash without publishing the raw mechanism.
2. Read bounded, time-identified 1D source samples.
3. Record raw and normalized inputs and the change between them.
4. Audit fixed-temperature, fixed-volume labels at selected tolerances.
5. Keep all augmented descendants within their source split.
6. Compare target coordinates using common physical source-term metrics.
7. Open the separate 2D test only after model selection.
8. Verify the installed runtime ABI before any coupled-CFD experiment.

All checks in this note were read-only except downloading small source-code
files and the mechanism to ignored local `runs/sources/`, and writing this
note. No simulation or model training was started by this investigation.

## Small copied 1D CVODE baseline: feasibility check

The existing case can support a small **restart experiment**, without copying
large datasets or rebuilding DeepFlame. Start from its reconstructed 0.0025 s
state. Do not claim this reproduces the original transient from ignition.
This section records inspection only; no solver was run on a case.

### Why not start from the old `0` directory?

The source `0` directory is not a complete, safe initial condition:

- `p` and `U` are absent; only `p.orig` and `U.orig` remain.
- Its file `C` is a `volVectorField` of cell-centre coordinates, not the
  required scalar atomic-carbon mass fraction. Blind copying would introduce
  a field-name collision.

In contrast, `${ONE_D_CASE}/0.0025` has all 62 required fields: `T`, `p`, `U`,
and the mechanism's 59 species. Each has the expected scalar/vector class.
No required field contains a code stream, coded boundary, or file include.
The reconstructed velocity has a fixed inlet value `(0.170038 0 0)` and a
zero-gradient outlet. Pressure is initially uniform at 101466 Pa, with a
zero-gradient inlet and a `waveTransmissive` outlet. These observations come
from the actual restart field files, not the nominal paper inlet pressure.

The five existing `constant/polyMesh` files are sufficient to preserve the
500-cell mesh. They are approximately 0.15 MB in total. Its patches are
`boundary` (empty, 2000 faces), `inlet` (one face), and `outlet` (one face).
Together with the restart fields and mechanism, the copied case is small.

### Explicit copy set

Create a new run directory under this project's ignored run tree. Copy only:

```text
copied-case/
  Okafor2018_s59r356.yaml
  0.0025/
    T, p, U, and each of the 59 named species
  constant/
    polyMesh/{points,faces,owner,neighbour,boundary}
    g
    thermophysicalProperties
    turbulenceProperties
    combustionProperties
    CanteraTorchProperties
  system/
    fvSchemes
    fvSolution
    controlDict
```

Use an explicit species-name list from the pinned mechanism, not a wildcard
copy of the source restart directory. Do not copy old `rho`, `phi`, `Qdot`,
enthalpy, or diagnostic fields; the solver can construct them from the
retained state. Do not copy `processor*`, `dynamicCode`, `postProcessing`,
old logs, or the source `Allrun`. Hash the copied inputs and record their
source locations in an ignored manifest.

The original `Allrun` invokes MPI using four subdomains from
`decomposeParDict`. That is unnecessary for a 500-cell serial smoke test.
The existing mesh also removes the need for `blockMesh`, `setFields`, or
`decomposePar` on the source case.

### Changes needed in the copy

1. Replace `controlDict` with explicit values: `startFrom startTime`,
   `startTime 0.0025`, `endTime 0.0026`, `deltaT 1e-6`, and
   `adjustTimeStep off`. This specifies 100 steps. Use `functions {}` and
   no `#calc`, `#include`, or sampling functions for the first smoke test.
2. Set output precision to 17 digits. Write only bounded checkpoints, such
   as every 25 steps, with no purge. Preserve the copied restart itself.
3. In `CanteraTorchProperties`, keep `chemistry on`, `transportModel Mix`,
   `inertSpecie AR`, and `splittingStrategy off`. Set `torch`, `GPU`, and
   torch logging off. Set load balancing off for the serial test.
4. Preserve original CVODE tolerances `1e-6/1e-10` for a first compatibility
   baseline. A tighter-reference comparison is a separate recorded variant,
   not an unrecorded change to the old case.
5. Add `h` and `hFinal` linear-solver entries. The old `fvSolution` only
   includes `ha` in its velocity/energy pattern, whereas the installed solver
   calls `EEqn.solve("h")`. The supplied image examples use `(U|h|k|epsilon)`.
   Existing `div(phi,h)` in `fvSchemes` already covers the energy flux.

The source `thermophysicalProperties` is intentionally an empty dictionary
after its header. The installed image's CH4 example does the same; its
Cantera mixture reads `CanteraTorchProperties`. Do not invent an unrelated
OpenFOAM thermophysical model to fill it.

### Pinned runtime checks

Image ID prefix `41bd0c7147ab` was inspected using temporary containers with
a read-only root filesystem and no network. The normal image entrypoint
loads OpenFOAM 7, activates the existing `deepflame` Conda environment, and
loads the DeepFlame environment. `dfLowMachFoam -help` completes successfully.
Its linked libraries have no unresolved entries in the inspected `ldd`
output. This is a startup check, not a successful CFD-run result.

The installed Python Cantera is **2.6.0**, and the executable links to
`libcantera.so.2` from that environment. The new labeling environment uses
Cantera 3.2.0. The installed environment has Python 3.8.20 and PyTorch
2.4.1+cu121. Record both versions and later compare selected local chemistry
updates; do not upgrade the working solver installation merely to make the
version strings equal.

The installed `createFields.H` selects sensible enthalpy, `hs`. The energy
equation adds chemistry heat release once. The installed bridge also divides
pressure by 101325 before ANN input packing. These agree with the source-level
contract above and confirm that the old `ha` solver settings need adaptation.

### Execution boundary and remaining checks

The implementing agent can first run `checkMesh` on the copied case, then a
one-step CVODE check, before attempting the 100-step restart. Mount only the
new project run directory writable; keep original assets read-only or omit
them from the container after copying. Use one CPU, bounded memory, no GPU,
no network, and a wall-clock timeout. Use the normal image entrypoint and
call `dfLowMachFoam -case <copied-case> -noFunctionObjects` directly.

Remaining uncertainties are actual boundary-condition compatibility at
startup, runtime for this 59-species case, and the effect of source-state
rounding and the Cantera version difference. A 100-step restart can establish
that a copied CFD case runs and produces bounded fields. It cannot establish
grid convergence, long-time flame stability, or learned-model reliability.
