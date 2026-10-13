# Scalar output calibration: an optional research family

Status: source check only. This does not select a new campaign or claim a gain.
Question: can a small per-species map of the frozen base output correct systematic
bias more safely than one species-wide constant? No additional state information
is supplied by such a map.

## Primary-source findings

Gupta et al. use monotone one-dimensional piecewise-linear calibration functions.
They fix knot locations at feature quantiles and learn knot values. Ordered knot
values impose monotonicity through linear inequalities. Their calibration is
input preprocessing jointly fitted with a lattice, rather than this project's
post-processing of physical increments. Joint calibration/lattice training can
be non-convex. The transferable part is the compact function representation and
explicit shape constraints, not an established chemical-accuracy result.
[Gupta et al., JMLR 2016, Sections 7.1 and 9.3](https://www.jmlr.org/papers/volume17/15-243/15-243.pdf)

Berta et al. define isotonic regression as ordered fitted values minimizing
weighted squared error. Pool-adjacent-violators produces constant blocks with
weighted means. They study its relation to binary probability calibration and
cross-entropy, and warn that calibration alone is a weak accuracy requirement.
These probability and scoring-rule results do not establish tolerance compliance
for signed physical increments. Absolute-error isotonic fitting would also be a
different objective from counting values inside specified acceptance intervals.
[Berta, Bach and Jordan, AISTATS 2024, Sections 1–2](https://proceedings.mlr.press/v238/berta24a/berta24a.pdf)

## Exact connection to the physical contract: project deductions

Let z=asinh(b/s), s=1e-14, and let f_i be a learned scalar map for species i.
Inference can use dhat=s*sinh(f_i(z)). Identity f_i(z)=z is the unchanged base.
The equivalent coordinate correction is f_i(z)-z. The deployed artifact contains
only knots and fitted values, not training-reference arrays or query labels.

For each training reference d, define tau=1e-15+0.1*abs(d) and
L=asinh((d-tau)/s), U=asinh((d+tau)/s). Then physical acceptance is exactly
L <= f_i(z) <= U in real arithmetic; numerical decoding must still be checked.
Maximizing the count of covered intervals is not least-squares isotonic regression.
For example, squared loss on asinh(d/s) targets their centers, although every
value inside [L,U] is scientifically acceptable. A convex distance-to-interval
penalty is feasible with fixed knots but is only a surrogate for covered count.
Do not call either objective an exact coverage optimizer without proving that.

Monotonicity is an assumption about base-output ordering, not a chemical law.
If two states have exactly the same base scalar and disjoint acceptable intervals,
no deterministic scalar map can satisfy both. Different chemistry can collapse to
the same scalar. A nondecreasing map also cannot undo arbitrary ordering errors.
Pinning f_i(0)=0 prevents opposite-sign corrections: a positive base remains
nonnegative and a negative base remains nonpositive. Therefore do not add a zero
pin merely for convenience when sign repair is part of the objective. Without
that pin a monotone map can shift its zero crossing, but still cannot route by
the unknown reference. These are structural limits, not measured dataset claims.

## Training-only check before choosing this family

Use existing permitted training rows. Freeze base identities and row ordering.
In a small, prespecified set of base-coordinate quantile bins, report counts,
base acceptance, target-coordinate spread, and acceptable-interval locations
relative to identity. Separate species and signs; record snapshot consistency.
A repeatable conditional displacement would support scalar bias correction.
Broad or contradictory intervals at similar base outputs would instead indicate
information that a scalar calibrator lacks. Bin summaries alone do not prove a
monotone fit exists or place a bound on every continuous scalar function.

If selected, preregister one compact knot count and objective, identity
initialization, duplicate-knot handling, constraints and extrapolation. Derive
knots from training outputs only. Any constant-map comparator must use the same
rows, objective and physical decoder; historical negative constants are not
automatically a matched control. Do not sweep losses or pick a species-wise recipe
using development results. A training-only split may check stability, but disclose
that the frozen base already saw these rows; it is not independent confirmation.

Report all inference operations, including endpoint extrapolation and sign changes.
Monotone maps do not guarantee pass preservation. Use full protected population
scores, repair/damage, signs, error tails and total latency before qualification.
This low-capacity family can test output-dependent bias; it cannot establish that
missing thermochemical information or arbitrary sign errors have been repaired.
