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

## Verification

```bash
python benchmarks/precision_conditioning/learning/prepare.py --dry-run
python -m pytest tests/test_precision_learning.py -q
python benchmarks/precision_conditioning/learning/train.py runs/representation/run-001 --dry-run
```

Expected: 384 planned intervals, disjoint parent groups and 12 model fits. Fast tests
check exclusion, leakage, train-only fitting, masked metrics and matched initialization.
The dataset and code remain inspectable; published HTML and figures are the review interface.
