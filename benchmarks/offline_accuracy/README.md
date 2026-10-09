# Offline representation accuracy

## Responsibility and non-goals

Compare four species-increment representations at matched model/update cost.
This is an offline chemistry study, not CFD transfer, a new integrator, or a
claim that CVODE tolerances bound every neural prediction. Original solvers,
environments, data, and prior results stay unchanged.

## Frozen first-stage contract

- Fixed-temperature/volume NH3/CH4 chemistry, 59 species, interval 1e-6 s.
- Existing six 1D flame snapshots: four training, two offline evaluation groups.
  Split before interpolation/perturbation. These are correlated snapshots from
  one realization, not independent flame cases or an unseen CFD test.
- T and composition sampling retain the documented flame augmentation. Pressure
  is sampled independently and uniformly over 0.95–1.05 atm (96258.75–106391.25 Pa).
  Its input coordinate has fixed centre 101325 Pa and scale 5066.25 Pa. Other
  input statistics and target scales use the selected training rows only.
- First matched sizes: nested 2000 and 10000 accepted rows where available;
  always report actual counts. Same evaluation labels at each size.
- Four targets: transformed-state, direct signed-power, state-budget-linear,
  and scaled asinh. The linear coordinate retains its prior state-budget
  definition; do not mislabel it as an oracle increment-dependent scale.
- Four 800-unit GELU layers, FP32, L1 transformed loss, batch 256, 2000 updates.
  Fixed final checkpoint, no evaluation-based checkpoint selection. Two fixed
  seeds 20261011 and 20261012; retain both, never choose the better seed.
- Primary physical increment budget: abs(pred-reference) <= 1e-15 + 0.1*abs(reference).
  The 10% level is an initial research target, not high precision or deployment
  approval. A 1e-12 increment may have about 1e-13 absolute error.
- Report a predeclared grid: absolute floors 1e-12, 1e-15, 1e-18 and relative
  tolerances 1, 0.1, 0.01, 0.001. The primary pair stays fixed; no threshold is
  selected because it makes a result look good. SSPI uses threshold 1e-15.
- Nominal whole-state milestone: 99% of evaluated states pass every non-argon
  species at the primary pair. This is an empirical research target, not a
  guarantee. Do not certify this milestone from subset-only reference checks.
- Show component/whole-state pass counts, per-species and magnitude-bin scores,
  SSPI and zero baseline, mass/element/positivity diagnostics, training process
  and wall seconds, and repeated inference timings. An empty group is null.
- Reference uncertainty is evaluated separately on the fixed audited subset.
  Require estimated uncertainty <=10% of the requested component budget for
  qualified subset scores. Unknown references remain unknown, not passes.
  Whole-population scores remain explicitly nominal unless all labels are checked.

## Interfaces and boundaries

`dataset.json` works with `flame_conditioning.prepare` and saves fresh immutable
states, lineage, signed CVODE labels and exclusions. `learning.json` and its fixed
second-seed variant work with `flame_conditioning.train`. New outputs belong in
ignored `runs/offline-accuracy/` in the existing project. Inputs are read-only.
No secrets, original mechanism content, or weights enter Git or public reports.

Minimal preparation preview:

```sh
python -m benchmarks.flame_conditioning.prepare <source> --config benchmarks/offline_accuracy/dataset.json --output <new-dataset> --dry-run
```

Verify with `python -m pytest tests/test_offline_accuracy*.py tests/test_flame_augmentation.py tests/test_flame_training.py -q`.
Expected: pressure bounds and prefix/split identity hold; old defaults remain
unchanged; tolerance counts, zero-baseline limitations, and unknown references
are handled explicitly. Model replay must match saved predictions before review.

Dependencies: existing flame augmentation, reference chemistry, model trainer,
stable coordinates and saved-artifact verification. Outputs feed the existing
GitHub review report. No production module depends on this experiment.

Start bounded runs only after dry-run/tests and reference audit. At most two
half-CPU jobs, 4 GiB each, in the existing immutable research image. No CFD run.
Increase to 50k/200k only after this cost/accuracy evidence; do not silently
replace these exploratory training limits with a paper-scale campaign.
