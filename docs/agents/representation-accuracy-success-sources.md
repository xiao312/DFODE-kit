# Representation accuracy: source check and proposed measures

Checked 2026-10-09. This note does not change a benchmark or start an experiment.

## Paper facts

Ke Xiao et al. report approximately 1.25 million original flame states and
8 million filtered, perturbed training states. Their pressure augmentation is

\[
p'=p+0.15[\max(p)-\min(p)]X,\qquad X\sim U[-1,1].
\]

This is 15% of the sampled pressure **span**, not 15% of pressure. They also
interpolate flame states and perturb temperature and composition.
[Sections 2.2–2.3](https://arxiv.org/html/2507.08277v2#S2.SS3)

The Small-Scale Prediction Index uses threshold \(\tau=10^{-15}\):

\[
\operatorname{SSPI}(\tau)=
\frac{\#\{i:|d_i|<\tau\ \land\ |\hat d_i|<\tau\}}
{\#\{i:|d_i|<\tau\}}.
\]

Here \(d\) denotes the species-increment target, to avoid confusion with an
endpoint mass fraction. The paper's Equation 8 uses \(Y\) notation. Test SSPI is
0.6900 for BC and 0.9874 for PT. BC has lower overall MAE and RMSE, and a higher
fraction within 1% relative error. Thus the metric changes the ranking.
[Section 3.2, Equation 8 and Table 1](https://arxiv.org/html/2507.08277v2#S3.SS2)

## What SSPI does not establish

The following conclusions follow from the formula, not from a new experiment:

- An always-zero predictor has SSPI 1 whenever the denominator is nonzero.
- Two targets below the threshold can differ greatly in relative terms.
- Two signed values below the threshold can differ by almost \(2\tau\).
- A value just outside the threshold can fail SSPI despite a small absolute error.

Use SSPI to ask, “Does a small true increment produce a small prediction?”
Do not use it alone to claim accurate nonzero increments or numerical-solver
accuracy. Include the zero predictor and report how many targets enter the metric.

## What CVODE tolerances mean

CVODE uses solution-dependent weights
\(W_i=1/(\operatorname{atol}_i+\operatorname{rtol}|y_i|)\) and a weighted
root-mean-square norm. It estimates local integration error and rejects a step
when its error test fails. This is not a guarantee of a specified number of
correct digits in every component or of bounded error over a full trajectory.
An RMS condition also differs from requiring every component to pass.
[SUNDIALS mathematical description, Equation 3.6 and local error test](https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html)

Cantera 3.2.0 sets ReactorNet state defaults to `rtol=1e-9`, `atol=1e-15`.
These are integration settings, not automatically suitable neural-network
acceptance criteria.
[Versioned ReactorNet source](https://github.com/Cantera/cantera/blob/v3.2.0/include/cantera/zeroD/ReactorNet.h)

## Proposed offline success contract

These are proposed definitions, not requirements from either source.

Separate the research question about increments from the application question
about the state after one chemistry interval. For reference increment \(d_i\),
prediction \(\hat d_i\), and reference endpoint \(Y_i^+\), report both:

\[
E_i^{\Delta}=
\frac{|\hat d_i-d_i|}{a_i^{\Delta}+r^{\Delta}|d_i|},\qquad
E_i^{Y}=
\frac{|\hat d_i-d_i|}{a_i^Y+r^Y|Y_i^+|}.
\]

The first measures increment accuracy. The second measures state error from
the same initial state, before any additional endpoint-rounding error. It is
inspired by solver weights but is not CVODE's internal local-error estimator.
For each selected contract, a component passes when \(E_i\leq1\).

Before training, fix the tolerance pairs, mechanism, chemistry interval,
state distribution, and data split. Report component pass fractions and the
fraction of states where **all** components pass. Also report per-species and
magnitude-bin results, tail errors, cost, and the zero baseline. A weighted RMS
score can be an additional solver-style summary; do not replace the worst
component with it without saying so.

Keep reference uncertainty separate. A convergence difference is an uncertainty
estimate, not a rigorous bound. If a justified bound \(u_i\) is available, an
error budget \(b_i\) supports a conservative pass only when
\(|\hat d_i-d_i^{ref}|+u_i\leq b_i\). Otherwise flag labels whose uncertainty is
too large for the requested test; do not silently count them as accurate zeros.

Use offline chemistry tests for this first decision. CFD transfer and coupled
trajectory tests answer later questions. Broader pressure sampling must use a
declared physical range and fresh labels at the perturbed pressure; it does not
require a CFD test before the representation comparison can proceed.
