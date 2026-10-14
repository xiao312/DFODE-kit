# Species-specific correction heads

Responsibility: test parameter sharing with near-equal total parameters. Not a
temperature switch, new physical model, loss experiment or promoted predictor.

`gradient_probe.run --experiment EXP-20261013-species-heads --project <root>
--output <new-run>` discovers; `--execute` starts admitted GPU work. The JSON
manifest defines all rows, seeds, work and architecture. Artifacts: fit config,
weights, histories, state-only predictions, timings and protected recount.

Dependencies: controller -> staging/worker guard; worker -> correction_scaling
trainer with a model factory; model -> standard GELU/affine operations and stable
asinh decoder; review -> protected scorer and independent arithmetic checks.
Production has no dependents. Reuse kinetic_features staging row identities,
but mount only base/preprocessing and learned weights at inference; no kinetics,
labels, secrets, network or home. Training sees only20k fitting labels.

Run `python -m pytest tests/test_species_heads.py -q`. Expect group/independent
forward parity, parameter-gradient separation, zero output and exact counts.
Then run `python -m benchmarks.offline_accuracy.paper_baseline.species_heads.review
--project <root> --run <run>`; all saved states and physical results are checked.
