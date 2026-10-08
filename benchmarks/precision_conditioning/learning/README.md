# Controlled representation test

## Responsibility and non-goals

Compare three target coordinates on a small checked H2/CH4 dataset. Reuse the
checkpoint 02 reference methods. This is an interpolation feasibility test, not
a full dataset, chemistry-solver replacement, or proof of high relative precision.
No residual model or trajectory rollout is included at this gate.

## Steps and interface

1. `prepare.py --output <new-run> --dry-run` prints the fixed data plan. Omit
   `--dry-run` to run CVODES parent trajectories and nine checks per interval.
2. `train.py <run> --dry-run` checks data fitness and prints the model plan.
   Omit `--dry-run` to run the fixed comparison. It refuses existing training output.
3. `review.py <run>` exports selected figures and a step-by-step HTML report.
4. `verify_run.py <run>` independently recomputes saved test scores, checks matching
   updates/initial weights, and replays each saved model without training or writes.

`experiment.json` owns the eight temperatures and fixed whole-parent split:
six train, one validation, one test per mechanism. Pressure is 1 atm and phi is 1.
There are eight anchors and three intervals per parent: 384 interval records total.
CVODES supplies starting states. Tight direct Radau supplies working delta labels.
All five reference checks must exist. Exclude a whole interval if any temperature
or species component fails the existing 1%-of-budget agreement check. Save excluded
IDs and reasons. This excludes difficult cases and must be disclosed in the report.

`dataset.py` validates record uniqueness, lineage, finite values, solver availability
and whole-trajectory split separation. Saved FP64 arrays include inputs, physical
changes, budgets, relative masks and IDs. No imputed or clipped labels are allowed.
Each training split must have at least 32 accepted intervals, validation and test at
least 8 each. Otherwise stop; do not change thresholds to force training to proceed.

The model predicts delta-T and delta-Y. Inputs are T, P, Y and log10(h). Fit input
mean/RMS, output RMS and asinh scale only on accepted training data. Each coordinate
uses identical 64/64 tanh networks, Adam 1e-3, 200 epochs, batch size 256 and seed
20261008. All models receive the same sample order and initial weights. Train for
the full fixed budget; select the saved epoch using validation species budget p99.
The test split never selects epochs, scales, or hyperparameters. Decode in FP64.
FP32 means normalized features, targets and network arithmetic are FP32; the input
normalization itself and raw reference data remain FP64. This differs from the
earlier physical initial-state re-integration test after raw FP32 rounding.
Channels that are exactly zero in every accepted training label are fixed to zero
for every variant; record those channel indices. No test labels choose these channels.
Input ranges within 64 FP64 epsilons of their own magnitude use magnitude scaling,
not tiny standard-deviation scaling. This prevents constant pressure and inert
fractions from becoming unit-scale roundoff noise. The rule uses training rows only
and does not discard small trace species that have real relative variation.

Report pooled species budget errors, separate reliable relative errors, temperature
errors, per-species and magnitude-bin metrics, negative endpoints, mass error,
training/inference cost, selected epoch and learning curves. A zero-change baseline
uses exactly the same test mask. Non-finite predictions fail a variant explicitly.
One test parent and one seed cannot establish robust generalization or a universal
ranking. Do not run full trajectories or residual learning solely because one
variant has the smallest error in this small test.

## Dependencies and boundaries

`prepare -> reference.pilot/chemistry -> reference.analysis`
`train -> dataset/reference.targets -> PyTorch`
`review -> saved dataset/training evidence -> existing Pages and issue attachments`

Use pinned reference requirements plus `torch==2.5.1+cpu` from the official CPU wheel
index. No production DFODE-kit training code depends on this module. All run state,
weights and arrays belong in ignored `runs/`. No credentials or network calls occur
inside the experiment. Install dependencies before launching the offline container.
Use one CPU, 4 GB memory, one thread and an external one-hour timeout. Each command
also records its elapsed time and source revision. Never overwrite a completed run.

## Training-fit diagnosis

`fit_probe.py <run> --mechanism h2 --target budget-linear` replays a saved FP64
model on training rows only. `--row 0` reduces the check to the first training row.
This read-only command exits 1 if any species error exceeds its physical budget.
It does not mean that the code is broken: it is a deliberately strict scientific
fit check. It never reads validation or test labels for scoring or selects a model.
Dependencies are the existing saved arrays, preprocessing, `train.network`, NumPy
and PyTorch. Outputs are JSON on stdout; there are no writes or network calls.
Expected for the checkpoint 03 models: exit 1, with a reported maximum above 1.

`fit_diagnostic.py <run> --output <new-fit-run> --dry-run` prints a bounded plan.
Omit `--dry-run` to run it. `fit_experiment.json` pins the plan. Use budget-linear
coordinates only, to separate output weighting from nonlinear inverse gradients.
Reuse preprocessing fitted on the original training split. No held-out rows enter
the diagnostic. Each mechanism uses three subsets: one sample, all eight accepted
anchors from the 1400 K parent at h=1e-6 s, and all accepted training intervals.
The one-sample subset is the first of those eight rows. Changing subsets is a
separate probe, not part of the matched loss comparison.

For each subset, initialize the same FP64 64/64 tanh model with seed 20261008.
Compare coordinate MSE with physical-budget MSE at fixed updates 200 and 5000.
Adam, learning rate 1e-3, full-batch inputs and output RMS scales stay unchanged.
If q is normalized output and s is its output RMS, the budget error is
`(q - q_reference) * s`. The physical loss averages its square over active
temperature/species channels, divided by one global constant `max(s_active)^2`.
This constant uses the original training data and preserves relative physical
weights. It changes Adam's effective epsilon; record it and do not claim that
different optimizers are mathematically equivalent. Neither loss directly
optimizes p99 or guarantees the maximum error. No best epoch is selected.

As a separate capacity control, solve only the final linear layer by NumPy SVD
least squares, with fixed initial hidden features, for the one/eight-row subsets.
Record rank and condition number. This is an interpolation/memorization control,
not evidence of generalization or a comparable-cost training method.
Record target round-trip error, all row IDs, source/data hashes, loss curves,
physical scores, final weights, preprocessing and predictions. The runner refuses
an existing destination, limits itself to one thread and one hour, and uses the
same offline one-CPU/4-GB container boundary as the original comparison.
`fit_review.py <fit-run>` makes a static report from those saved results.
`verify_fit.py <fit-run>` replays final weights, recomputes physical scores and
checks matching initialization, sample IDs and update counts. Expected: 12 Adam
fits and four linear-head controls. Scientific pass requires every species and
temperature component to be within budget; a failed pass is retained, not hidden.
Tests verify the physical loss/gradient identity and train-only subset selection.

```bash
python benchmarks/precision_conditioning/learning/fit_diagnostic.py runs/representation/checkpoint03-20261008 --output runs/representation/fit-001 --dry-run
python benchmarks/precision_conditioning/learning/fit_diagnostic.py runs/representation/checkpoint03-20261008 --output runs/representation/fit-001
python benchmarks/precision_conditioning/learning/verify_fit.py runs/representation/fit-001
python benchmarks/precision_conditioning/learning/fit_review.py runs/representation/fit-001
python -m pytest tests/test_precision_fit.py -q
```

Only the final model weights are saved. Curve samples retain species p99/max but
not intermediate temperature maxima. Do not call an intermediate species-only
pass an all-component pass. Fixed 200/5000 checkpoints retain complete scores.
The first run found transient species fits followed by accuracy loss, not proof
that Adam can never fit a single example. The diagnostic does not change or repair
the original models; its failed scientific thresholds remain part of the evidence.

## Verification commands

## Repeatable fit and optimizer polishing

`polish.py <checkpoint-03-run> --output <new-run> --dry-run` reads the saved
training arrays and prints the pinned `polish_experiment.json` plan. Without
`--dry-run`, it runs three seeds, two mechanisms, three subsets (one/eight/all
existing training rows), and three methods: 54 bounded fits. The all-row subset
is a bridge to data scaling, not a new acceptance requirement for the narrow gate.
It does not generate chemistry, use held-out rows, or change production training.

Each method starts from the same seed and at most 200 full-batch Adam updates at
1e-3. All use budget-linear targets, the existing training-only normalization,
FP64, 64/64 tanh layers and coordinate MSE. Only the refinement changes:

- Adam decay: retain warm-up moments; decrease 1e-3 to 1e-7 geometrically over
  at most 5000 further accepted updates.
- L-BFGS: start new optimizer state; at most 1000 steps, history 50, learning rate
  1, strong-Wolfe line search and zero gradient/change stopping tolerances. Record
  closure evaluations separately; these are not equal-compute methods.
- Linear-head control: freeze the warmed-up hidden features and solve the output
  layer once by SVD least squares. Record feature rank and condition number. This
  remains an interpolation control, not proof of a scalable training recipe.

Check the maximum physical error across ALL temperature/species components after
each accepted update (not line-search trial points). Stop immediately at <=0.5
budgets, save that model, and require <=1 on independent replay. The margin is a
predeclared safeguard, not a changed evaluation budget. Save the lowest-maximum
model if no step passes. Record both its selected step and the final attempted
step. No held-out selection is involved. Store compact all-component histories,
full selected metrics, row IDs, source/input hashes, weights and preprocessing.
Reject nonfinite values. Each fit has a 60-second wall limit; the entire run has
900 seconds, within the external one-hour/one-CPU/4-GB/no-network limit. Timeouts
retain completed evidence and do not count as successful fits.

The narrow milestone requires one method to pass independent replay on both the
one/eight-row subsets, both mechanisms and all three seeds (12/12). Report
iterative-optimizer and linear-head-control gates separately. A control-only pass
does not justify a large dataset. `verify_polish.py <run> --require-narrow <method>`
is read-only and exits nonzero if that scientific gate fails.

Dependencies: `polish -> polish_core -> train.preprocessing/network` and
`fit_diagnostic.training_data/subsets/physical_scores`; NumPy/PyTorch only at fit
time, with the existing Cantera import dependency. `polish_review.py` consumes only
saved results for the existing static report and figures. All state stays in
ignored `runs/`; no secrets or network operations occur in these modules.

```bash
python benchmarks/precision_conditioning/learning/polish.py runs/representation/checkpoint03-20261008 --output runs/representation/polish-001 --dry-run
python benchmarks/precision_conditioning/learning/polish.py runs/representation/checkpoint03-20261008 --output runs/representation/polish-001
python benchmarks/precision_conditioning/learning/verify_polish.py runs/representation/polish-001
python -m pytest tests/test_precision_polish.py -q
```

### Dataset expansion gates

Do not spend indefinitely on eight-point memorization. After this bounded test:

1. Check repeatable narrow fitting, then inspect the existing 133 H2 / 144 CH4
   training rows. A control-only pass requires a small generalization feasibility
   test, not immediate large-scale labeling.
2. Freeze the candidate recipe and split new independent parents before deriving
   intervals. Run a modest generalization pilot. Require finite predictions,
   measured physical errors, no unexplained conservation/positivity failure, and
   useful improvement over zero-change in BOTH p99 and exceedance across parents.
   This is a gate for studying scaling, not a solver-replacement acceptance test.
3. Propose approximately 1k -> 4k -> 16k checked intervals per mechanism, with a
   fixed validation set and a sealed new test set. Increase independent parents
   and targeted coverage, not only adjacent anchors. These are planning sizes,
   not authorized launches or guaranteed accepted counts. Use learning curves
   and measured data-generation/training costs to choose the next step.
4. Continue if more independent data improves held-out errors at practical cost.
   If training remains poor, address fit/capacity. If training is good but held-out
   error is poor, test coverage. If more data does not help, stop blind expansion.
   Full-budget accuracy across the entire chemistry domain is NOT required before
   studying data scaling. Slurm/approved compute placement is required for scale.

## Original comparison verification

```bash
python benchmarks/precision_conditioning/learning/prepare.py --dry-run
python -m pytest tests/test_precision_learning.py -q
python benchmarks/precision_conditioning/learning/train.py runs/representation/run-001 --dry-run
```

Expected: 384 planned intervals, disjoint parent groups and 12 model fits. Fast tests
check exclusion, leakage, train-only fitting, masked metrics and matched initialization.
The dataset and code remain inspectable; published HTML and figures are the review interface.
