# Align the precision study with CFD use

Status: research note, 2026-10-08. No new experiment is approved or started by this note.

## Decision

Keep the small reactor tests as numerical diagnostics. Do not require perfect fitting of every component in those tests before starting a representative flame benchmark. The next question is whether the chemistry update preserves flame behavior at useful cost.

## Verified paper facts

Source: Ke Xiao et al., [author manuscript, arXiv v2](https://arxiv.org/html/2507.08277v2). Section numbers below refer to this version, not a checked publisher PDF.

| Topic | Reported setup |
| --- | --- |
| Chemistry (Sections 1, 2.1) | Okafor: 59 species, 356 reactions. Map `(T,p,Y)` to species increments; Cantera/CVODE labels. |
| Sampling (2.2) | One premixed flame: 60% NH3/40% CH4, equivalence ratio 1, 300 K, 1 atm; 500 cells, 1 microsecond steps, 2.5 milliseconds; approximately 1.25 million states. |
| Augmentation (2.3) | Interpolate neighboring spatial states at uniform temperature intervals. Perturb temperature, pressure and species; filter temperature, N2 and heat release. Select approximately 8 million perturbed states. |
| Network (2.4) | Four 800-unit hidden layers; Z-score normalization; omit inert argon output. |
| Offline evaluation (3.2) | 48,916 test states from a 4-millisecond turbulent snapshot. Metrics include relative-error fractions and small-target classification at 1e-15. |
| CFD evaluation (2.5, 3.1, 3.3) | 1D profiles; 2D turbulent kernel, 28-mm square, 512-square grid. Compare flame area, heat release and species fields. |
| Operational policy (3.3) | Box-Cox above 1000 K; direct-power at 305–1000 K; zero increments below 305 K. |

Reported chemistry/total speedups: 526/20; CPU baseline versus CPU-plus-GPU inference, not equal hardware (Section 3.3). The paper favors different accuracy measures for appreciable and negligible changes (Section 2.4).

## What this means for our study

The completed tests answer useful but narrower questions: can we resolve labels, distinguish arithmetic loss from training error, and save an accurate fit? They do not test transport-conditioned inputs, repeated source-term errors, flame motion, or pollutant fields.

Increasing the number of homogeneous-reactor examples alone would not remove this mismatch. Also, failure under our strict all-component budget does not by itself prove failure in CFD. Conversely, a good average CFD result would not prove high relative precision in every tiny increment. These are separate claims and need separate evidence.

## Proposed next benchmark, not a reproduction claim

1. Pin the NH3/CH4 mechanism and one operating point. Verify species order, thermodynamic data, reactor constraint, chemistry interval and energy coupling against the runtime. Preserve a cheap installed-mechanism smoke test, but do not call it the application benchmark.
2. Sample cold gas, preheat region, flame front and burnt gas from a resolved 1D reference flame. Add controlled interpolation and perturbation. Record acceptance rates and the distribution before and after filtering. Do not silently label all endothermic states as invalid.
3. Label a bounded, stratified subset first. Use CVODE for routine labels. Keep independent increment checks on a smaller subset, with unresolved-label masks. Do not require the most expensive audit on every future row.
4. Start a nested data-size ladder on this representative source: for example, 10k, 50k and 200k accepted states. These sizes are proposed, not values from the paper. Report learning curves and cost. Keep model capacity, training compute and precision controlled or explicitly report their changes.
5. Split before augmentation. Keep related source states and their descendants together. Reserve independent flame realizations or CFD snapshots for evaluation. A random split of neighboring cells is not strong evidence of generalization.
6. Compare transformed-state increments, direct-power increments, budget-linear increments and scaled-asinh increments. Include the temperature-switched baseline explicitly; otherwise we are not comparing against the paper's operational approach.
7. Run a short 1D coupled simulation before a large turbulent case. Compare propagation, temperature, heat release, major species and selected nitrogen intermediates. Report conservation drift, negative states, failures and fallback use. Assess both global and temperature-conditioned errors.
8. Proceed to a separately budgeted 2D test only after the 1D result is stable. Report chemistry time, total time and hardware separately. Set application acceptance thresholds before comparing models.

## Details to resolve before implementation

Do not infer an exact reproduction from the manuscript alone. Confirm the mechanism file and checksum, precise blend basis, interpolation spacing, perturbation draws and normalization, heat-release cutoff, chemistry-step constraint, training schedule, snapshot selection, and boundary handling at switching temperatures. Check the publisher version and released case/code assets for differences. Define the runtime source-term and energy interface from code, not from the shorthand mapping above.

## Recommended immediate scope

Prepare and review the flame-state sampling and evaluation contract. Retain the existing precision harness as a supporting audit. Do not launch the original million-state dataset or turbulent simulation without a measured pilot cost and compute allocation.
