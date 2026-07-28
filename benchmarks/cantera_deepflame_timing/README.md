# Standalone DeepFlame-style Cantera timing harness

This C++ benchmark mirrors the core chemistry loop in DeepFlame's
`dfChemistryModel::solveSingle()` without running OpenFOAM:

1. Reuse one `Cantera::Reactor` and `Cantera::ReactorNet`.
2. Disable reactor energy integration.
3. Set each cell state with `(T, P, Y)`.
4. Call `syncState()`, `advance(deltaT)`, read mass fractions, and reset time.
5. Construct `RR_i = rho * (Y_i,next - Y_i,current) / deltaT` and `Qdot`.

It records separate timings for state assignment, reactor synchronization,
Cantera integration, state readback, network-time reset, and source-term
construction. It also records the public Cantera 2.6 CVODES counters available
through `Integrator`: RHS evaluations and final method order.

## Build on `lh40902`

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate df-share

cd /data1/kexiao/260624_ml4chem_ode_lh40902/DFODE-kit/benchmarks/cantera_deepflame_timing
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The executable initializes `CANTERA_DATA` from
`$CONDA_PREFIX/share/cantera/data` when the variable is unset. This is needed
for the `df-share` Cantera 2.6 Conda package, whose compiled fallback data path
contains trailing null bytes. Python masks the packaging defect by adding its
own package data directory, while a standalone C++ process otherwise fails
when loading `element-standard-entropies.yaml`.

## Run the DeepFlame H2 HIT operating point

```bash
./build/cantera_deepflame_timing \
  --mechanism /home/dfode/deepflame/deepflame-dev-d1bde9a/examples/dfLowMachFoam/pytorch/twoD_HIT_flame/H2/Burke2012_s9r23.yaml \
  --fuel H2:1 \
  --oxidizer O2:1,N2:3.76 \
  --temperature 300 \
  --pressure 101325 \
  --phi 1 \
  --dt 1e-7,1e-6,1e-5 \
  --states 11 \
  --repeats 5 \
  --warmup 1 \
  --rtol 1e-6 \
  --atol 1e-10 \
  --profile-advance \
  --output-dir /data1/kexiao/260624_ml4chem_ode_lh40902/260713_cpp_cantera_timing_burke
```

Generated files:

- `summary.csv`: aggregate timing distribution and phase shares for each `dt`.
- `cell_timings.csv`: one row per measured synthetic cell call.
- `rhs_evaluations.csv`: one row per CVODES RHS callback when
  `--profile-advance` is enabled.
- `metadata.txt`: mechanism, operating condition, tolerance, and build metadata.

`--profile-advance` uses timed `ReactorNet::eval()` and `Reactor::eval()`
overrides to decompose `ReactorNet::advance()` into:

1. Reactor equation and chemistry RHS evaluation.
2. ReactorNet callback overhead outside the reactor.
3. Remaining CVODES work, including nonlinear iteration, error control, linear
   algebra, and final interpolation.

The raw RHS log includes the solver time requested by every callback, which
reveals repeated trial states and expensive regions. Callback profiling adds
two clocks and an in-memory record per RHS evaluation, so use an unprofiled run
for final speed comparisons. Cantera 2.6 does not publicly expose accepted
step, error-test, nonlinear-iteration, or Jacobian counters; those require a
patched 2.6 integrator or Cantera 3.2 `solverStats()`.

When compiled against Cantera 3.2 or newer, the harness automatically writes
`solver_stats.csv` using the native `ReactorNet::solverStats()` API. It records
accepted steps, direct and linear-solver RHS evaluations, Jacobian evaluations,
linear setups, nonlinear iterations and failures, and error-test failures for
every measured cell. Comparing `rhs_evals + lin_rhs_evals` with
`profiled_rhs_calls` validates the callback profiler independently.

Use `--no-cell-csv` for production-scale timing where detailed file output
would perturb the measurement. The synthetic states interpolate from cold
reactants to constant-enthalpy, constant-pressure equilibrium products. They
are useful for timing coverage but are not a substitute for replaying states
sampled from an actual CFD field.
