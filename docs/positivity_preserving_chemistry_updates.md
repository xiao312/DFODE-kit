# Positivity-preserving chemistry updates

DFODE-kit provides four complementary positivity routes for fixed-interval
chemistry prediction. Physical composition, reaction matrices, corrections,
and diagnostics remain in float64. Neural features and hidden layers remain in
float32.

## Sandu composition projection

The original composition-space correction solves

\[
\min_Y \frac{1}{2}\lVert Y-\widehat Y\rVert_2^2
\quad\text{subject to}\quad
A Y=A Y_n,\quad Y\geq Y_{\min}.
\]

`A` contains independent elemental and total-mass constraints. This gives the
smallest Euclidean correction that is both positive and conservative.

## Reaction-space Sandu projection

The reaction-coordinate counterpart solves

\[
\min_\xi \frac{1}{2}\lVert\xi-\widehat\xi\rVert_2^2
\quad\text{subject to}\quad
Y_n+S_m\xi\geq Y_{\min}.
\]

Because every update remains in `Range(S_m)`, conservation is structural. The
metric differs from composition projection: it prefers the smallest reaction
trajectory correction rather than the smallest species correction.

## Relative projection

For trace species, absolute Euclidean distance gives major species too much
weight. The relative projection uses

\[
W=\operatorname{diag}(1/q_i),\qquad D=W^{-2}
\]

and applies

\[
\delta_c=\delta-D A^T(A D A^T)^+A\delta.
\]

The implementation iterates positivity flooring and weighted correction, with
the exact Sandu projection as a final fallback.

## Neural process-Patankar model

The network predicts nonnegative forward and reverse reaction demands. For
each species, all competing consumption demands share a depletion ratio. Each
reaction process receives the most restrictive ratio of its reactants. Hence
total consumption cannot exceed the initial available amount:

\[
\sum_r C_{ir}\xi_r\leq Y_i^n.
\]

The update

\[
Y^{n+1}=Y^n+(P-C)\xi
\]

is nonnegative and stoichiometrically conservative by construction. This is a
first-order process-based neural Patankar layer; it avoids claiming that the
general nonlinear multi-reactant PMPRK conjecture has been proved.

## Reaction-trajectory free-energy model

An SVD of the molar stoichiometric matrix supplies a reduced independent basis
`B`. The network predicts:

- an initial reduced reaction trajectory;
- positive reaction mobilities;
- a diagonal preconditioner.

The fixed proximal iterations operate on

\[
c=c_n+B\eta.
\]

The local convex Gibbs functional is chosen so its gradient at the initial
state exactly matches the Cantera reaction affinities in `Range(S)`. A
fraction-to-boundary step keeps `c` positive. A final backtrack enforces

\[
G_{\mathrm{local}}(c_{n+1})\leq G_{\mathrm{local}}(c_n).
\]

This gives positivity, atom conservation, and local thermodynamic monotonicity
in one structural update. It is a practical reduced proximal approximation,
not a claim that a general irreversible GRI-Mech system satisfies the
detailed-balance theorem in Liu, Wang, and Wang.

## Commands

Evaluate a baseline with a projection:

```bash
dfode-kit evaluate-positive-interval \
  --checkpoint baseline.pt \
  --source validation.h5 \
  --mech mechanism.yaml \
  --method sandu-composition \
  --output sandu-composition.json
```

Train a structural model:

```bash
dfode-kit train-positive-interval \
  --variant neural-patankar \
  --train-source train.h5 \
  --validation-source validation.h5 \
  --mech mechanism.yaml \
  --output neural-patankar.pt
```

## References

- Sandu, *Positive Numerical Integration Methods for Chemical Kinetic Systems*.
- Kircher and Votsmeier, atom-balance and positivity layers for kinetic
  surrogates.
- Modified Patankar and process-based modified Patankar Runge-Kutta methods.
- Liu, Wang, and Wang, reaction-trajectory variational schemes for
  detailed-balance reaction systems.

## Sandia GRI30 matched-data result

All trainable variants below used the same random train/validation split,
fixed `dt = 1e-7 s`, GRI30 mechanism, signed-power `alpha = 0.1`, MAE, and
combined transformed `Y_next + 0.1 delta_Y` objective. Trainable models ran
for 300 epochs.

| Method | Species MAE | SSPI 1e-15 | SSPI 1e-12 | T MAE | Negative entries | Mean element residual | GPU us/sample |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing baseline | 3.813e-5 | 0.9775 | 0.9807 | 2.494 K | 18.68% | 1.08e-17 | 22.7 |
| Sandu composition | 3.813e-5 | 0.9784 | 0.9796 | 2.494 K | 0 | 1.12e-17 | 42.4 |
| Relative projection | 3.813e-5 | 0.9787 | 0.9808 | 2.494 K | 0 | 5.36e-17 | 37.6 |
| Reaction-space Sandu | 3.813e-5 | 0.9790 | 0.9807 | 2.494 K | 0 | 1.08e-17 | 64.8 |
| Neural Patankar | 1.958e-5 | 0.9981 | 0.9981 | 2.610 K | 0 | 1.08e-17 | 31.0 |
| Reaction trajectory | 5.457e-5 | 0.9777 | 0.9807 | 2.523 K | 0 | 1.08e-17 | 50.4 |

The reaction-space projection required composition-space fallback for 823 of
11,164 validation rows because the original GRI30 reaction-coordinate metric
is rank-deficient and numerically ill-conditioned near trace-species
boundaries. This fallback is explicit in the JSON diagnostics.

The reaction-trajectory model produced mean local free-energy change
`-2.04e-13`, maximum `1.03e-14`, and zero changes above the reporting
tolerance `1e-12`. This is monotonic for the implemented affinity-consistent
local Gibbs functional, not a proof of monotonicity for the full changing-T/P
Cantera Gibbs energy.

The current practical baseline is Neural Patankar. It improves species
accuracy and SSPI while enforcing positivity and stoichiometric conservation
without a post-inference optimization. Its slightly worse temperature MAE
should be addressed with loss balancing rather than weakening the structural
species update.
