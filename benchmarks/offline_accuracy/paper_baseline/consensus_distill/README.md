# Bounded consensus compression

Responsibility: test whether one frozen base plus a small head can reproduce a
useful two-base restraint at lower cost. This is not new reference chemistry,
unconstrained correction learning, or proof of severe-error repair.

Inputs: registered JSON, both frozen base caches for training-only teacher
targets, fixed common50k IDs excluding both U/head-fit sets, and one base's800
last hidden activations. The head predicts a sigmoid fraction in[0,1]. Its
decoder moves only toward zero by at most min(0.1*abs(b),
0.1*(1e-15+0.1*abs(b))). Both sign and magnitude stay protected. The scale is
part of the predictor, not the scoring allowance.

Controls: a per-species mean teacher fraction, fitted on the same50k only, and
the unchanged one-base predictor. No label-dependent routes at inference.
Use the existing correction trainer with explicit prediction/initial-check
callbacks; historical defaults remain unchanged. The loss is soft-target BCE
on a bounded teacher fraction, not a physical accuracy claim.

Security: trusted staging holds references. Training sees only declared50k
states/labels and own frozen base. Fresh inference sees states/IDs, own frozen
base, trained head/constant fractions. It does not see the second base, labels,
audit answers, network, home or credentials. GPU admission and isolation remain
in gradient_probe.run and hidden_features.run_worker. No production dependent.

Example: `python -m benchmarks.offline_accuracy.paper_baseline.gradient_probe.run
--project . --experiment EXP-20261013-consensus-distill --output <new-run>`;
add `--execute` after tests. Outputs include selected IDs, targets, head weights,
teacher-fit history, isolated physical predictions, runtime and protected review.

Verify `python -m pytest tests/test_consensus_distill.py -q` on the pinned GPU
environment. Expected: decoder bounds/signs, teacher reconstruction, soft-target
loss, parameter count, frozen-base parity and exact saved inference replay.
