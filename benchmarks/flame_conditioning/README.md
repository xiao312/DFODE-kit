# Flame-conditioned chemistry benchmark

## Responsibility and non-goals

Test target representations on states from the NH3/CH4 flame application.
Keep the earlier homogeneous-reactor precision tests as independent diagnostics.
Do not require their strict training-fit gate before this experiment.
This module does not change production training, installed solvers, original study
cases, or shared environments. A one-step test is not coupled CFD validation.

## Plan and decision gates

1. Inspect existing study cases and mechanisms read-only. Record hashes and the
   actual chemistry interface before extracting any data.
2. Extract a bounded set of 1D reference-flame states. Keep cold, preheat, reaction,
   and burnt regions. Split source groups before augmentation. Record lineage.
3. Label with the installed CFD chemistry constraint: closed constant-volume,
   fixed-temperature ideal gas, over 1e-6 seconds. Predict species increments only.
   This differs deliberately from the old adiabatic constant-pressure audit.
4. Check CVODE tolerance convergence and a direct-increment Radau subset. Report
   unresolved increments; do not replace a failed label by zero.
5. Compare transformed-state, signed-power, budget-linear and scaled-asinh targets
   at matched architecture, seed, inputs and update budget. Evaluate the paper's
   temperature-switched BC/PT policy separately. Fit scales on training data only.
6. Measure errors on held-out CFD states, by temperature and target magnitude.
   Only then attempt a short coupled 1D test in a copied case. A stable reference
   run must precede any model-in-the-loop run. Do not claim a 2D validation from
   an offline 2D snapshot test.

Start with a cost/coverage scout, then a bounded training set. Proposed later
sizes are 10k, 50k and 200k accepted states, not automatic launches. Increase size
only after measuring cost, physical failures, and useful validation improvement.
Keep the test set sealed until the comparison recipe is fixed. Never choose
hyperparameters from test results. Correlated snapshots are not independent runs.

## Inputs, outputs and configuration

`chemistry.py` supplies explicit state validation, isothermal fixed-volume CVODE
endpoints, an independent direct species-increment RHS, and conservation checks.
Input state: `T` (K), `P` (Pa), `Y` (mechanism-ordered mass fractions). No silent
normalization or clipping is permitted. A caller must record any preprocessing.
Output: signed species increments, temperature/pressure/density changes, solver
statistics and elapsed time. Direct integration keeps initial density and
temperature fixed, not pressure. Pressure can change when molecular count changes.

The CFD source interface observed in the installed solver is
`RR_i = rho_initial * delta_Y_i / interval`; its heat release uses formation
enthalpies. The CFD energy equation remains outside this local chemistry solve.
Record the exact runtime revision and species order before model deployment.

All generated states, copied cases, model weights, logs and reports go into a new
directory beneath ignored `runs/flame-conditioning/`. Refuse existing output
destinations. A mechanism copied from user study assets is a provenance-backed
working candidate, not an independently verified publisher supplement. Do not
redistribute its contents until its license and source are established.

## Dependencies and dependents

`coordinates.py` implements the four targets. The transformed-state baseline uses
stable `log1p`/`expm1` differences and inverses so avoidable subtraction error is not
mistaken for a learning defect. Inputs use `Y**0.1`, affine-equivalent to Box-Cox
after standardization, with linear T and p. Training data alone set means/scales.
A predicted negative Box-Cox power base has no valid inverse: set that endpoint
to zero and report the correction mask separately. Do not silently call corrected
predictions unconstrained. Other coordinates do not impose positivity. Keep raw
negative endpoint and conservation metrics for all models before any CFD repair.

Dependency graph: `case extraction -> saved states -> chemistry -> checked labels
-> representation training -> independent evaluation -> static review`.
Numerical dependencies are pinned by `../precision_conditioning/reference/requirements.txt`.
Training additionally uses the existing pinned CPU PyTorch environment.
No production module depends on this benchmark. Existing GitHub review surfaces
consume small static evidence only, not raw case files or credentials.

## Security and execution boundary

Read existing cases and solver assets through read-only mounts. Do not run their
Allrun/Allclean scripts in place. Do not install into shared environments, rebuild
an existing image, terminate another process, or expose a network service.
Use the existing image by immutable digest and the project-owned environment.
Each live pilot has no network, one CPU at most, 4 GB memory, one numerical thread,
and an external timeout of at most one hour. Use lower limits when the host is busy.
Slurm is required on HPC login nodes; an image does not bypass placement rules.

## Verification and minimal example

`extract.py --case <read-only-study-case> --mechanism <study.yaml> --output
<new-run>/source --dry-run` reads six paired 1D line snapshots and checks their
column/coordinate alignment. Omit `--dry-run` to save original and explicitly
normalized states, time/coordinate lineage, source hashes and a private mechanism
copy. The command never writes to the original case. `--times` selects exact source
directory names. All selected snapshots are from one realization; no claim of
independent-flame validation follows from this extraction.

`scout.py <source-directory> --output <new-run>/scout --dry-run` chooses 32
distinct states spanning temperature, using no labels for selection. Omit
`--dry-run` for five CVODE settings and two direct-increment Radau settings per
state. The default wall limit is 900 seconds, with append-only raw records and
incremental summaries. The original-study tolerance is included for comparison,
not presumed accurate. Budget and increment-relative fitness are separate
diagnostics. They do not imply that the text-source states have exact digits.

`prepare.py <source-directory> --output <new-dataset> --dry-run` validates
`dataset.json`. Omit `--dry-run` to create 10k training and 1024 validation states.
Split snapshots before augmentation; interpolate only neighboring spatial samples.
Draw target temperature uniformly, perturb T by up to 100 K and each species
exponent independently by up to 0.15. Preserve interpolated argon, normalize other
species, and filter N2 to the source range padded by 5%. Pressure is interpolated
without random perturbation. No heat-release filter is applied. These explicit
choices differ from an exact paper reproduction. Save lineage and acceptance rates.
Use tight fixed-T/V CVODE labels. Reject nonfinite/negative endpoints or failed
conservation/constraint checks, with row-specific reasons. Save partial arrays
every 100 rows and never substitute zero for missing labels. Validation snapshots
are correlated parts of the same flame; a separate 2D test is still required.
This runner never reads test states. Its default 2400-second limit fits a
one-hour external timeout. Audit selected augmented labels independently before
training; the unperturbed scout alone does not certify augmented labels.
The dataset runner reuses a parsed mechanism and reactor, but resets state, time
and solver history for each row. Tests compare this path with fresh reactors.

`audit_labels.py <dataset> --output <new-audit> --dry-run` selects 16 states per
split by temperature alone. The live run compares stored labels against fresh
CVODE, step-limited CVODE and two direct Radau calculations. It records cancellation
and the original-study tolerance comparison. Training requires this selected subset
to pass the empirical 1%-of-budget agreement check. This is not certification of
all rows. The audit has a 900-second internal deadline and never opens the 2D test.

`data.py` refuses incomplete artifacts, checksum/species/split mismatches and more
than 10% excluded labels. It exposes training/validation data only. `metrics.py`
reports non-argon component errors, small-target retention at 1e-15, temperature
and magnitude bins, per-species errors, inverse-domain corrections, negative
endpoints, mass/element drift and source heat-release errors. Relative errors on
ordinary labels are explicitly nominal, not individually solver-certified.
The heat-release calculation follows `-rho/interval * sum(hf298_i * delta_Y_i)`;
it is not a learned temperature endpoint or full CFD energy solution.

`train.py <dataset> --audit <passing-audit> --output <new-training-run> --dry-run` validates the dataset
and `learning.json`. After an augmented reference audit, omit `--dry-run` for
16 fits: four coordinates, FP32/FP64, and nested 2k/10k requested training sizes.
Excluded labels reduce the larger actual size; record both counts. Use identical
128/128/128 tanh networks, initialization and minibatch sampling order. Each fit
has 2000 Adam updates, batch 256, cosine learning-rate decay from 1e-3 to 1e-5,
and a 600-second per-variant limit inside a 3000-second run limit. A timed-out fit
is explicitly incomplete and not a matched-budget result. Validation every 100
updates selects the lowest non-argon physical-budget p99. Correction and physical
failure rates remain visible; selection alone is not a deployment decision.
Argon is fixed, not learned. Train-only means/standard deviations and asinh scales
are saved with model weights, selected training indices, validation predictions,
curves and source hashes. No 2D test is read. This smaller network is a bounded
feasibility model, not the paper's four-by-800 network or matched paper compute.
The runner checks that the passing augmented audit references the exact dataset
manifest hash; a stale or failed audit cannot authorize this comparison.

```bash
python -m pytest tests/test_flame_*.py -q
```

Expected: valid states are preserved; invalid states fail; fixed temperature and
density are maintained; the independent RHS conserves mass/elements; endpoint and
direct increments agree on a small installed-mechanism test. These are fast
correctness checks, not a claim of final NH3/CH4 label accuracy.

The source/runtime evidence is recorded in
`docs/agents/flame-source-and-runtime-contract.md`. The user approved this route
for work through 08:00 China time on 2026-10-09. Save a reviewable checkpoint before
that deadline; do not launch work that cannot be bounded by it.
