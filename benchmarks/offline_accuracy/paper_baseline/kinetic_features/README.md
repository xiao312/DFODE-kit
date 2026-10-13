# Initial kinetic features

## Responsibility and non-goals

Test whether known initial-rate structure improves correction fitting and transfer.
This is a registered training-pool diagnostic, not an integrator or promotion.
Reuse existing chemistry setters, frozen-rate formula, correction trainer and
protected scorer. Do not change tolerances or introduce endpoint corrections.

## Interface and configuration

`gradient_probe.run --experiment EXP-20261013-kinetic-features` admits the job;
discovery is read-only without `--execute`. The experiment JSON defines seeds,
roles, arms, widths, updates and finite fit time. `prepare` selects exactly the
previous 50k diagnostic rows, keeps 20k fitting labels separate, and stages only
states for inference. `worker` fits fresh heads; `infer` reloads state-only models;
`review` independently scores saved predictions and validates identities.

Inputs: pinned base/preprocessing, private mechanism, original FP64 T/P/Y and
permitted fitting labels. Outputs: frozen configs, scaler, checkpoints, histories,
full signed predictions, timing and recount. Large files stay in ignored runs.

## Dependencies and boundary

`gradient_probe.run -> kinetic_features.prepare/execute -> existing guarded worker`.
`worker -> correction_scaling.train`; `model -> physics_prior.frozen_step +
chemistry.set_state + correction_scaling.model`; `review -> protected scorer`.
Production has no dependency on this module. CPU kinetics uses one thread;
learning requires GPU. Training sees only its 20k labels. Replay has state-only
input and model artifacts; no reference arrays, credentials, home or network.

## Verification

Run `python -m pytest tests/test_kinetic_features.py -q` in the pinned server
environment. Expect scaler, shape, rate units/parity, no-normalization, zero
correction and parameter-count tests to pass. Use the registered two-seed runner
then `python -m benchmarks.offline_accuracy.paper_baseline.kinetic_features.review
--project <root> --run <run>` for raw-array counts and provenance checks.
Live timing includes rate computation; cached features are not a latency claim.
