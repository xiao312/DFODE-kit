# Frozen-base hidden-feature comparison

## Responsibility and non-goals

Test feature access against near-equal head capacity. Keep the continued GBCT base,
10k rows, Smooth L1 coordinate loss, fixed-asinh decoder and final checkpoint.
No new labels, loss, CFD, split changes, manual routing or incumbent replacement.
The governing gates are in `research/`, not in this candidate module.

## Declared comparison

A: 119 inputs, widths 256/256/256 (177210 parameters).
B: 119 inputs plus 800 post-final-GELU frozen activations, widths 256/256/256
(382010 parameters). Standardize extra activations on the fixed 50k calibration
prefix; replace zero standard deviation by 1. No label enters this transform.
C: 119 inputs, widths 800/256/256 (381754 parameters).

Rerun all three arms with seeds 20261011/20261012. Use zero output initialization,
FP32 heads, FP64 loss/decoding, Adam cosine 1e-4 to 1e-6, 6000 updates, batch10000,
60M presentations. No clipping or checkpoint selection. The owner removed time
caps; each fit still has finite declared work. Compare measured time separately.
Training progress uses training labels only; final scoring is outside the worker.

## Inputs, outputs and security

`prepare` reads hash-checked base/caches/selections, computes frozen hidden scales
and stages only 10k training labels. Evaluation input files contain states and
ordered IDs only. Inference receives only model artifacts and those state files.
`worker` fits or predicts in a restricted Bubblewrap process. Source, environment,
input and model mounts are read-only; only a new output directory is writable.
Network, home, full project runs and reference labels are absent from inference.
An explicit canary checks label invisibility and read-only mounts before launch.

The controller remains trusted and has access to labels. This isolates candidate
processes, not the interactive operator. Independent confirmation needs separate
custody and is disabled. No credentials, sockets or other users' files are mounted.

## Dependencies and verification

`prepare -> correction_scaling readers/model + hidden model`;
`worker -> correction_scaling.train/objectives + hidden model`;
`controller -> research_contract admission/sandbox/scorer + worker`.
The historical trainer accepts absent development diagnostics; historical callers
retain their original behavior. No production module depends on this experiment.

Example: `python -m benchmarks.offline_accuracy.paper_baseline.hidden_features.campaign
--project <root> --output <new-directory>` validates; `--execute` admits work only
after M0, hash checks, sandbox probes and idle-GPU checks. One fit at a time.

Run `python -m pytest tests/test_hidden_features.py tests/test_research_contract*.py -q`.
Expected: parameter counts, zero-output identity, feature shapes, decoder replay,
absence of label-dependent inference, fail-closed promotion and sandbox admission.
Validate cached versus state-only inference exactly on all three populations.
