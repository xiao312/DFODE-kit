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
every 100–1000 rows and never substitute zero for missing labels. Validation snapshots
are correlated parts of the same flame; a separate 2D test is still required.
This runner never reads test states. Its default 2400-second limit fits a
one-hour external timeout. Audit selected augmented labels independently before
training; the unperturbed scout alone does not certify augmented labels.
The dataset runner reuses a parsed mechanism and reactor, but resets state, time
and solver history for each row. Tests compare this path with fresh reactors.
If a dataset reaches its wall limit, `--resume-source <stopped-dataset>` continues
into a new output directory with identical source/configuration. It rechecks saved
hashes, preserves completed rows and failure records, and records the prior run's
manifest hash. The stopped artifact is never edited. The separate continuation
has a 900-second limit by default (`--continuation-wall-seconds`, at most 3600).

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
`verify_scaling.py <small-dataset> <large-dataset>` is read-only by default. It
checks source/config identity, the raw training prefix, identical validation
states and accepted masks, and label differences against the same error budget.
`--output <new.json>` saves those checks for the review. Require a pass before
treating the two training runs as a controlled data-size comparison.

The next optional density step is `dataset-200k.json` with
`learning-source-longer200k.json`. Start it only after the full 50k comparison
passes replay and its conventional baseline improves both validation budget p99
and heat-release RMS over the matched 10k run. Keep source snapshots, validation,
seed, architecture, updates, batch size and learning-rate schedule unchanged.
This tests data density, not broader flame coverage. Equal updates imply fewer
average presentations per state in the larger set. Use the tested new-output
continuation path if labeling reaches its one-hour segment limit. Preserve all
segments; never extend a live limit or overwrite a stopped run. Check prefix and
validation identity, and independently audit the final data before fitting.
Full-array compression uses 100–1000-row checkpoints, recorded per split, while
progress still prints every 100 rows. This bounds repeated serialization cost
without changing sampling, chemistry, acceptance, or final saved arrays. A
graceful wall stop always writes the final checkpoint. Resume accepts only such
a completed time-limit checkpoint with verified hashes, not an arbitrary crash.
Do not launch a final fit after 06:30 China time on 2026-10-09; leave time for
checks and the 08:00 checkpoint. An unfinished larger run is partial evidence,
not grounds to delay or relabel the completed comparisons. Keep the 2D test
sealed until this pre-test decision and all included model identities are fixed.

Before opening the test, `learning-density-repeat.json` repeats only the
conventional 10k/50k density comparison with fixed training seed 20261010. It
keeps architecture, update count, batch size, learning rates, and validation
selection unchanged. This checks sensitivity to initialization/minibatch order;
it is not a new four-target ranking, a seed search, or a reason to select the
best seed. Keep both repeated fits in the frozen test list. The primary 200k
comparison still uses seed 20261009. Run this bounded repeat only as the second
half-CPU job beside label generation, not as an added third workload.

Before either 200k model result or the reserved test is available, extend the
same conventional-only seed check with `learning-density-repeat200k.json`.
It changes only requested training count and the descriptive selection text.
Require the completed 200k dataset's prefix check and passing reference audit.
Keep seed 20261010, model, batch size, updates, learning rates, and selection rule
unchanged. This is one additional fit, not a seed search. Start it only before
06:30 China time and within the two-job/half-CPU-per-job limit. Retain its result
even if it disagrees with the primary seed, and freeze it with the other models
before opening the 2D source. Report partial work explicitly if it cannot finish.

## Reserved 2D snapshot evaluation

`diagnose_test_labels.py <failed-test> --output <new-diagnosis> --dry-run`
checks a completed test-label artifact without loading or scoring models. The live
run profiles strict endpoint rejections, then checks up to eight failed states:
the first failure, the most negative endpoint, and fixed temperature quantiles.
Compare the saved reused-reactor solve with a fresh reactor, a tighter absolute
tolerance, a step-limited solve, and two direct-increment Radau solves. Save all
signed values and hashes. The output is a diagnosis, not a replacement label set
or an automatic change to the acceptance rule. Require a new output directory;
the failed test and frozen models remain read-only. Dependencies are the existing
chemistry module and pinned research environment. Verify with
`python -m pytest tests/test_flame_label_diagnosis.py -q`.

### Reference-policy amendment before first 2D model score

The original strict endpoint rule rejected 789/1216 selected 2D cells, all for
negative values of magnitude at most 2.084e-36. Most were cold cells. Fresh
reactors reproduced the saved increments; tighter CVODE and direct Radau checks
on seven failed states agreed within 2.856e-9 of the physical budget. Strict
sign rejection therefore removes valid numerical information non-uniformly.
The original test, strict acceptance mask, frozen models, and failed evaluator
remain unchanged. No model score was produced before this amendment.

`recover_test_reference.py <strict-test> --output <new-test> --dry-run` previews
an explicit, separate numerical-reference policy. The live path rechecks every
previously rejected cell with fresh CVODE, tighter CVODE (absolute 1e-24), and
two direct Radau settings. Only sign-only failures can be reconsidered. Require
finite values, the existing mass/element/fixed-T/V constraints, disagreement at
most 1% of the physical budget, and negative endpoint magnitude no greater than
the original solver absolute tolerance (1e-21) in every check. The latter is a
numerical-noise screen, not a rigorous positivity or accuracy guarantee. See
[SUNDIALS advice on negative values](https://sundials.readthedocs.io/en/v7.5.0/cvode/Usage/#advice-on-controlling-unphysical-negative-values).

Do not alter any signed increment or source cell. Preserve the old mask and
recovery flags in `reference-quality.npz`, full comparison records in JSONL, and
both old and new artifact hashes. This changes reference eligibility only, not
models, sample membership, normalization, scoring, or temperature thresholds.
Keep unresolved near-zero components separate from precision claims. A fresh
subset audit and the unchanged evaluator gate must still pass on the new test.
Report this as a post-reference/pre-score amendment, not the original strict
protocol. Do not use it to silently reclassify old training data. The internal
wall limit is 1800 seconds; partial recovery cannot be scored. Verify with
`python -m pytest tests/test_flame_reference_recovery.py -q`.
`verify_reference_recovery.py <strict-test> <recovered-test> --output <new.json>`
independently checks the unchanged cells, frozen plan, signed increments, old
acceptance mask, all per-row comparison records, and amended eligibility masks.
Require its pass before the fresh subset audit and model scoring. It does not
change either artifact. The report includes its compact, hash-bound evidence.

`historical.py` reads only the numerical arrays from `checkpoint_conversion/`.
It never reads the old pickle files or runs the old training scripts. Both controls
use the saved 4x800 GELU FP32 weights, original species order and Pa pressure.
We predeclare two reconstruction modes: `source-formula` retains FP32 output
arithmetic (and the conventional script's reconstructed initial endpoint);
`stable-adapter` denormalizes in FP64 and reconstructs changes from the actual
input state. The latter explicitly records conventional inverse-domain repairs.
Neither mode applies the old CFD wrapper's pressure overwrite or mass repair.
The source formula records invalid conventional power bases without hiding them.
These are historical-weight controls, not a paper reproduction. Power statistics
were saved only in FP32. The original BC training sources overlap this 1D domain;
historical 2D use is not excluded. Do not call either control a fresh held-out test.
`historical_validation.py <dataset> --audit <audit> --historical <converted-dir>
[--historical <second-dir>] --output <new-dir> --dry-run` checks these inputs.
Omit `--dry-run` for a small offline inference test in the unchanged research
runtime. It saves both controls and the predeclared fixed-temperature hybrid.
The reserved-snapshot command also accepts repeated `--historical` arguments;
it freezes their array/manifest hashes and both reconstruction modes before
opening the snapshot. No original model is selected from test scores.

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
Record wall and process time for one offline batch prediction per model, excluding
model loading, reference generation, and metric calculation. These timings include
the adapter and reconstruction. They are not CFD speedups: host contention, CPU
quota, batch size, communication, and solver coupling can change deployment cost.
`verify_heldout.py <test> <evaluation>` is a read-only reconciliation of the saved
cell IDs, population masks, prediction arrays, and main physical scores. It checks
that all frozen model and hybrid identities are present, with no extra models.
It uses the independent Cantera-density calculation from `verify_physical.py`.
`--output <new.json>` saves the verification record. This is a metric check, not
an additional reference-accuracy or coupled-CFD claim.

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
After that check, `--tolerance-preset tight` creates a separate copy with CVODE
relative/absolute tolerances 1e-12/1e-21. The default `study` preset is unchanged.
Both copies keep ANN disabled and the same fixed timestep, mesh and initial fields.
`compare_cfd.py <study-case> <tight-case> --original <read-only-source>` checks
both completed runs, matching input hashes and the one allowed dictionary change.
It reports final field differences; `--output <new.json>` saves the result.
Default is read-only. This is tolerance sensitivity over 100 steps, not a rigorous
error bound, timestep/mesh convergence, or a neural-model accuracy/speedup test.
The installed solver also constructs a spray cloud. The copy uses the repository's
inactive `sprayCloudProperties` dictionary, records its hash, and rejects a template
that does not explicitly disable both spray activity and coupling. The original
study case has no such file. A failed first startup is retained as evidence.

`review_cfd.py <copied-case> --original <read-only-original-case> --output <new.json>`
checks completion, finite scalar fields, mass closure and the final temperature.
It rechecks every allowlisted original file against its pre-copy hash. These
checks establish startup/restart compatibility and preservation of the source,
not a validated flame speed, mesh convergence, or neural-model accuracy.
The temperature profile uses actual nonuniform mesh coordinates, checked against
the old geometry-vector `0/C`. That file is read as geometry only and is never
copied into the carbon-species field of a new case.

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
The authored report source and its runtime boundary are in `report-content/`.
`--parity <runtime-parity.json>` includes the unchanged-runtime comparison.
`--cfd-tolerance <comparison.json>` includes the separate 100-step numerical
tolerance control. Its final-state budget score is not a one-step model score.
When refreshing an existing app snapshot, preserve its stable ID and title.
The held-out evaluator and both review commands accept `--dry-run` to validate
their inputs and destination without writing an output artifact.
`verify_physical.py <dataset> --training <run> [--training <run>] [--historical
<validation-run>]` independently recomputes the main validation claims from saved
predictions. It checks sample identities, budget p99, negative endpoints, mass
drift, selected NH3/CH4/NO/OH species p99, and heat-source RMS using Cantera density rather than the training metric's
density formula. It is read-only by default; `--output <new.json>` saves evidence.
It also checks saved temperature-bin budget p99 and target-magnitude-bin absolute
error p99, including exact counts and empty-bin nulls. These bins are fixed before
test scoring. The report preserves both test populations separately.
This does not replace saved-model replay or independent reference integration.
`verify.py <dataset> <training> --training-metrics --output <new.json>` also
replays all selected training states and independently checks their physical
scores. Use it before interpreting a training/validation gap. Without the flag,
the faster existing validation replay remains unchanged.
The report compiler also accepts `--historical-validation`, `--heldout`,
`--scaling`, `--expanded-audit`, and repeated `--filter-audit` completed JSON
evidence. Filter rows retain split counts and the exact rule; they do not modify
the dataset or introduce a new model-selection rule. Historical controls
remain separate from equal-budget new fits, and the two test populations remain
separate. It never publishes weights or raw mechanism files.
`--heldout-audit <summary.json>` adds the test's independent subset check and
must match the audit hash in the completed held-out evaluation. Keep numerical
zero references separate from unresolved nonzero entries. Test sampling rows
report selected, accepted, and excluded counts for each overlapping population.
`--reference-recovery <verification.json>` adds the explicit amendment evidence.
`review_recovery.py` requires its test hash to match the scored test and keeps
strict and recovered counts separate; no raw private state is published.

## Post-score input-support diagnosis

`input_support.py <training-dataset> <completed-primary-run> <test-dataset>
--output <new.json> --dry-run` checks provenance and the common input scaler of
the four primary models. The live path reports raw training ranges, encoded
feature scales, and validation/uniform-test/balanced-test ranges and standardized
distances. These populations remain separate. It does not load weights, rerun
predictions, change preprocessing, or train a model. All inputs are read-only;
the only output is a new small JSON file. Dependencies are the existing dataset
reader and coordinate implementation. A diagnostic range violation is not a
causal attribution or proof of multivariate support. This analysis occurs after
test scoring; any later model repair needs fresh test evidence. Verify with
`python -m pytest tests/test_flame_input_support.py -q`.

`filter_audit.py <dataset> [--training <completed-run>] --output <new.json>`
measures the effect of the inspected historical curation rule without changing
data: keep `sum(hf_298 * delta_Y) <= 200 J/kg`. This is not a universal validity
test and does not mean rejecting every endothermic step. Report accepted-label
counts and temperature segments before interpreting sensitivity scores. Optional
model scores are validation-only diagnostics on kept/rejected subsets, not a new
model-selection rule or permission to filter the reserved CFD test. `--dry-run`
checks input provenance without writing. Existing labels and results are immutable.

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
