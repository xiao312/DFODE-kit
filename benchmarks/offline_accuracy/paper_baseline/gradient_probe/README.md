# Continued-base gradient diagnostic

## Responsibility

Measure whether repair and preservation losses conflict in the existing base's
parameter space, before another optimizer or architecture experiment. This is
not a fitted candidate, a promotion, or evidence of held-out generalization.

Use eight fixed disjoint 1,024-state batches from the authorized 50k training
population, per seed. The next batch in cyclic order is a separate training-pool
check for each one-step probe. No U/D or confirmation labels enter this worker.
The original weights are restored between probes and at completion.

## Measurements

Use the existing differentiable GBCT reconstruction and physical log objective.
Fix groups from the incumbent's saved prediction: all, base-pass, base-fail,
floor, weak nontrivial and active increments. Compute group parameter gradients,
norms and pairwise cosines for all parameters, trunk and final affine layer.
Compare the direct all-group gradient with the count-weighted pass/fail sum.
Output gradients for disjoint component masks are not an interference test.

Then take one actual fresh-Adam step at learning rate 1e-6 on the all-component
loss. Record group loss changes and physical pass/repair/damage counts on the
same batch and the next fixed batch. Reset all model weights before each probe.
This does not approximate the original optimizer's momentum state. Compare the
differentiable reconstruction with the saved deployment predictions and report
any discrepancy in physical tolerance units before interpreting gradients.

The separately registered `EXP-20261013-step-scale` repeats the same diagnostic
at 1e-7 and 1e-8 learning rates, with original weights restored before each
step. The existing 1e-6 result remains the control; it is not rerun. Use
`--experiment EXP-20261013-step-scale` and a new output directory. This tests
finite-step amplitude, not another objective or a promoted predictor.

## Interface, dependencies and boundary

`python -m benchmarks.offline_accuracy.paper_baseline.gradient_probe.run
--project <root> --output <new-dir>` performs discovery; `--execute` admits the
registered diagnostic. Outputs are selected training IDs, group statistics,
finite-step effects, runtime/source identities and hash-bound artifacts.
`run -> research_contract admission/sandbox + hidden_features.run_worker`;
`worker -> existing base loader, input features and differentiable decoder`.
No production dependent exists. Training labels are permitted; evaluation data,
secrets, network and host home are not mounted. Use one idle GPU and one CPU thread.

## Verification

Run `python -m pytest tests/test_gradient_probe.py -q`. Expect known conflicting,
orthogonal and zero-gradient cases and a correctly normalized group sum. Check
restored weights byte-for-byte in memory. Candidate selection and any future
training require a new manifest; negative cosine alone does not authorize PCGrad.
See `research/experiments/gradient-interference-source-note.md` for the narrow
primary-source basis and limits of this adaptation.
