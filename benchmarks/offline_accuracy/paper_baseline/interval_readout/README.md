# Frozen GBCT interval readout

Learn one affine correction to the original normalized GBCT output using frozen
base hidden features. Compare point versus exact acceptance-interval supervision.
This is not the earlier asinh correction, a reference-based inference selector,
or permission to change the physical evaluator.

Inputs: pinned common50k training states/references, base, preprocessing, and
registered config. Outputs: paired heads, training histories, state-only full
population predictions, timings and independent physical scorecards. Both arms
use the same finite component width even where the acceptance set is unbounded.

Dependencies: existing guarded controller, Bubblewrap, correction_scaling trainer,
fixed-batch base network/decoder and protected evaluator. No production dependent.
Training gets only training references; inference mounts only states and fitted
artifacts. No credentials/network/confirmation data enter workers.

Run through `gradient_probe.run --experiment EXP-20261013-interval-readout
--project . --output <new-run>`; add `--execute` after read-only checks.
Review with `interval_readout.review --project . --run <run>` as a module.
Verify `pytest tests/test_interval_readout.py -q` in the pinned GPU environment.
Expected: exact interval membership, saturated lower branch, paired scales and
zero correction. Full replay and scientific qualification require server runs.
