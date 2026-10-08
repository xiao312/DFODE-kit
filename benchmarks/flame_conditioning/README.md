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

`verify.py <dataset> <training-run>` is read-only. It checks all planned fits,
dataset hashes, nested training indices, matched initialization, complete update
budgets, replayed validation predictions and independently recomputed budget p99.
Expected: 16 verified models for the default plan. Verification of saved evidence
is not a scientific accuracy pass. `load_predictor` exposes the same explicit
decode/correction interface for later held-out and copied-case evaluation.

The first 16-model run completed and replayed successfully, but large training
and validation errors remain. FP32/FP64 results nearly overlap. The bounded
follow-up `learning-longer.json` changes only update budget (20k), validation
cadence (500 updates), and keeps FP32/the full accepted 10k candidate set. It
repeats all four targets with the same 128/128/128 model. This is an optimization
budget check, not proof that data size is irrelevant or a broad parameter search.

`learning-source-backbone.json` checks the inspected study's four 800-unit GELU
layers and normalized L1 objective. It uses our grouped splits and train-only
statistics. Its 2000 updates and 256-row batches are much smaller than historical
training. This is a backbone/loss control, not an exact paper reproduction or an
isolated activation ablation. Optional `activation` and `loss` fields default to
`tanh` and `mse`, preserving previous configurations and saved-model replay.

The initial backbone control lowered validation p99 for transformed-state and
budget-linear targets, but did not meet the strict error budget. The next bounded
scaling check uses `dataset-50k.json` and `learning-source-longer.json`: 10k updates
on the available 10k or 50k candidates, four targets, identical architecture and
compute. The requested size is capped by availability; compare actual counts.
The sampling seed makes the original 10k raw states a prefix of the 50k set and
keeps validation states identical. Verify these facts before comparing results.
Audit the expanded training labels separately. The 2D test remains sealed until
these predeclared comparisons finish; do not tune from its later scores.

## Reserved 2D snapshot evaluation

`heldout.py --source-array <reserved.npy> --mechanism <study.yaml> --training
<completed-training> [--training <another-completed-training>] --output <new-test>
--dry-run` checks a frozen comparison plan. The live command records model and
preprocessing hashes before reading the test array. Source columns are T, p, and
mechanism-ordered Y, verified against named original fields during discovery.
Draw 1024 cells uniformly without replacement plus up to 32 cells per fixed
temperature bin. Save cell IDs and separate membership masks. Report uniform-cell
metrics separately from the diagnostic temperature-balanced sample; do not pool
them into a domain-average claim. Normalize only the small recorded mass closure
error (at most 1e-4), never negative fractions. Keep excluded labels and reasons.
An independent `scout.py` audit on this artifact must pass before test scoring.

`evaluate_heldout.py <test> --audit <test-scout> --output <new-evaluation>` rechecks
the frozen hashes and scores every listed model without tuning. It also evaluates
the fixed paper-style policy: zero below 305 K, direct power from 305 to 1000 K,
and transformed-state increments at or above 1000 K. The thresholds are not fit.
This is an offline test on one 2D snapshot, not temporal rollout or coupled CFD
validation. Test results cannot select further tuning on this same test set.

## Copied CFD restart

`copy_case.py --source-case <original-1D-case> --mechanism <study.yaml> --output
<new-case>` is a read-only preview. Add `--apply` to copy an explicit allowlist:
the reconstructed 0.0025 fields, five mesh files and selected dictionaries. It
refuses existing output and does not run the solver. The source `0` directory is
not usable: it lacks U/p and has a coordinate-vector C rather than carbon Y.
The copy adds the installed solver's h/hFinal settings, disables ANN/GPU/load
balancing/functions, and uses literal fixed-step control values. Default is one
step; `--steps 100` gives the separately bounded 100-step compatibility run.
Keep original CVODE tolerances 1e-6/1e-10 for this first runtime check. The installed
Cantera is 2.6.0, unlike labeling 3.2.0; do not upgrade it. A copied restart smoke
test is neither an ignition reproduction nor a learned-model CFD validation.
The installed solver also constructs a spray cloud. The copy uses the repository's
inactive `sprayCloudProperties` dictionary, records its hash, and rejects a template
that does not explicitly disable both spray activity and coupling. The original
study case has no such file. A failed first startup is retained as evidence.

`review_cfd.py <copied-case> --original <read-only-original-case> --output <new.json>`
checks completion, finite scalar fields, mass closure and the final temperature.
It rechecks every allowlisted original file against its pre-copy hash. These
checks establish startup/restart compatibility and preservation of the source,
not a validated flame speed, mesh convergence, or neural-model accuracy.

`runtime_parity.py <dataset> --output <new.json> --dry-run` checks a bounded
32-state validation subset in the installed CFD Python/Cantera environment.
Omit `--dry-run` to compare tight fixed-T/V increments with the saved research
labels. This Python 3.8-compatible runner does not import training code, install
packages, or load checkpoints. It tests version/interface agreement, not all
states or a rigorous reference bound. Keep version differences visible before
any model is considered for the older CFD runtime.

`review_snapshot.py --training <summary.json> [--training <summary.json>]
--audit <augmented-audit.json> --cfd <cfd-review.json> --output <reviewed.json>`
creates sanitized, source-backed rows for the GitHub Pages review. It reads small
saved evidence only. It does not train, alter results, or include private host
paths, raw chemistry states, model weights, or mechanism contents. The HTML report
is a downstream presentation of this snapshot, with the source hashes retained.

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
