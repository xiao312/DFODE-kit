# Reaction structure for fixed-step signed increments: source note

2026-10-13. Literature and design assessment only; no experiment, environment
change, mechanism modification or candidate success is claimed.

The proposal has credible architectural precedents. Their evidence primarily
concerns learning mechanisms, instantaneous source terms or reduced mechanisms;
it does not establish improved signed species increments at fixed temperature,
fixed volume and `h = 1e-6 s` under this repository's physical acceptance rule.

## What the primary papers establish

| Primary source | Architecture and evidence | Relevant limit |
| --- | --- | --- |
| Ji and Deng, *Autonomous Discovery of Unknown Reaction Pathways from Data by Chemical Reaction Neural Network* (2021), [paper](https://arxiv.org/pdf/2002.09062), DOI `10.1021/acs.jpca.0c09316` | Reaction neurons encode mass action and Arrhenius kinetics; output weights represent stoichiometry. CRNN-ODE fits concentration trajectories through an ODE solver (Eqs. 6–8). | A learned vector field integrated numerically is different from directly predicting an accurate finite-step increment. The original architecture does not enforce every elemental balance. |
| *Robust mechanism discovery with atom conserving chemical reaction neural networks* (2024), [paper](https://doi.org/10.1016/j.proci.2024.105507) | A dedicated layer restricts learned stoichiometries to atom-conserving combinations; original CRNN test cases show improved mechanism discovery and robustness to deficient data. | Conservation is a structural restriction, not evidence that each small species increment meets our tolerance. Here the mechanism is already known. |
| *Global reaction neural networks with embedded stoichiometry and thermodynamics for learning kinetics from reactor data* (2024), [paper](https://doi.org/10.1016/j.cej.2024.149863) | Positive latent forward rates feed a De Donder layer, `r_j = r_j+ (1 - Q_j/K_j)`, then a fixed stoichiometric source-term layer (Eqs. 4–5). Demonstrations use catalytic microkinetic source terms and reactor data. | Strong precedent for embedding affinity and equilibrium in a **rate** architecture. Global rates are not uniquely determined by species source terms; the paper explicitly discusses this. It does not validate our stiff fixed-step map or cost target. |
| Padiyar, Dash and Aditya, *A graph neural network based chemical mechanism reduction method for combustion applications* (2026 preprint), [paper](https://arxiv.org/html/2603.22318v1) | Message passing on species/reaction connectivity uses stoichiometry and kinetic features to predict species/reaction importance. Reduced mechanisms are evaluated using reactor/flame quantities. | This is mechanism reduction, not a direct increment predictor. Its use of instantaneous kinetics also carries a runtime cost. Treat it as a topology precedent, not finite-step accuracy evidence. |

## Exact structure, and what it does not determine

Let `N = nu_products - nu_reactants` have species rows and reaction columns.
For a closed, constant-volume system with concentrations `c`:

```text
dc/dt = N q(c, T)
xi_h = integral_0^h q(c(t), T) dt
Delta c = N xi_h
Delta Y = diag(W) N xi_h / rho
```

Here `xi_h` is an integrated net extent per volume, not an initial rate times
`h`. The last equation uses fixed mass density and molecular weights `W`.
Cantera exposes the separate stoichiometric arrays and reaction rates of
progress, with bulk rate units `kmol/m^3/s`; this fixes the units and indexing
of the construction. [Cantera kinetics API](https://cantera.org/stable/python/kinetics.html)

The following are algebraic implications, not reported experimental gains:

- If `N` has rank `r < n_reactions`, then `xi_h + z` gives the same endpoint
  whenever `N z = 0`. Endpoint labels cannot identify the physical extent of
  every reaction. A pseudoinverse supplies one coordinate convention, not the
  true reaction history. An orthonormal basis of `range(N)` removes this
  redundancy, but its coordinates cease to be individual reaction extents.
- If the elemental atom matrix is `E`, then `E N = 0`; the decoder conserves
  elements. It does not ensure `Y + Delta Y >= 0` or numerical accuracy.
  `range(N)` is contained in `ker(E)` but equality must be checked for the
  actual mechanism and inactive-species treatment.
- Small species increments can be differences of large reaction contributions;
  reversible net extents can additionally be differences of large forward and
  reverse extents. Relative accuracy of large latent outputs therefore need
  not yield adequate absolute accuracy of their sum. Assess decoded physical
  errors and floating-point cancellation, not only latent regression loss.

Project inspection supplied by the supervising worker (not a literature
result): the pinned Cantera 3.2 mechanism has 59 species, 356 reactions and
5 elements, with `rank(N) = 54`, `rank(E) = 5`, and `max(abs(E N)) = 0`.
Consequently `range(N) = ker(E)` numerically and the reaction-coordinate null
space has dimension 302. An unrestricted extent decoder adds no linear
restriction beyond elemental conservation for this mechanism. Its value, if
any, must come from how the model learns reaction coupling, not a smaller
admissible linear space. The mechanism contains 340 reversible reactions,
16 irreversible reactions, and 50 third-body reactions including 29 falloff
reactions. Mechanism SHA-256:
`26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0`.

## Thermodynamics and graph details that must remain explicit

Use full chemical potentials, not standard-state potentials alone:
`A_j = -(N^T mu)_j`. Cantera distinguishes composition-dependent
`delta_gibbs` from `delta_standard_gibbs`, and reports `chemical_potentials`
in `J/kmol`. A dimensionless affinity must use matching units for `R T`.
[Kinetics API](https://cantera.org/stable/python/kinetics.html),
[thermodynamic API](https://cantera.org/stable/python/thermo.html)

For a reversible, thermodynamically consistent rate law, the GRNN relation is
equivalently `q_j = q_j+ [1 - exp(-A_j/(R T))]`. This links the **instantaneous**
rate direction to current affinity. Our inference is that freezing `A_j(0)`
and multiplying by an arbitrary positive learned magnitude is not a justified
sign restriction on `xi_h`: coupled reactions change composition and affinity
during the step. Nor does an individual reaction's affinity determine a species
increment, which sums all reactions. A continuously dissipative rate field also
does not automatically make a one-step decoder positive or dissipative. At
fixed `T,V`, the relevant closed-system potential is Helmholtz free energy;
chemical potentials still provide its composition derivatives. These are
thermodynamic distinctions, not a new constraint validated on the dataset.

Preserve mechanism semantics. Cantera reactions have an explicit `reversible`
flag; irreversible fitted reactions must not silently receive reverse rates
from an equilibrium formula. Separate forward/reverse irreversible entries and
duplicate channels need separate identities until their equivalence is checked.
[Cantera kinetics API](https://cantera.org/stable/python/kinetics.html)

A graph built only from nonzero entries of `N` loses participants that occur on
both sides. In particular, third-body colliders can change kinetics while
their net stoichiometric coefficient is zero: Cantera uses
`[M] = sum_k epsilon_k c_k`. Preserve reactant and product incidence separately,
plus collider efficiencies and reaction type. Falloff and other
pressure-dependent rates also require their declared metadata; fixed `T,V`
does not imply fixed pressure as composition evolves. This recommendation
follows the documented rate dependencies; the counts above separately record
the project inspection. [Cantera reaction-rate definitions](https://cantera.org/stable/reference/kinetics/reaction-rates.html)

## Narrow next comparison

The existing [kinetic-feature trial](EXP-20261013-kinetic-features.json) and
[element-projection trial](EXP-20261013-element-projection.json), with outcomes
in [RESULTS](../RESULTS.md), already cover initial-rate feature augmentation and
post-hoc conservation projection. The new distinguishing hypothesis is whether
reaction connectivity improves **learning finite-map coupling**, since the
inspected stoichiometric range supplies no extra linear restriction.

Before fitting, complete a read-only audit of reference reconstruction in the
54-dimensional mass-weighted stoichiometric subspace, cancellation, and all
mechanism participant metadata. Do not create 356 supposedly unique extent
labels from a pseudoinverse. This note itself ran no numerical audit.

If that audit supports a trial, compare one reaction-graph encoder against a
dense encoder with the same available state/thermodynamic information and a
shared fixed orthonormal stoichiometric decoder. Prespecify comparable parameter
and training work budgets, identical finite-step labels, physical objective,
paired samples, seeds and final checkpoint policy. Include the unchanged
incumbent separately. Keep chemical-potential augmentation as a later factor,
or supply it equally to both arms; do not change graph, decoder, inputs and
loss simultaneously and attribute the outcome to connectivity.

Evaluate actual returned increments with unchanged acceptance, repair/damage,
sign, tail, positivity and full-cost guards, including feature construction.
A graph-only transfer gain would support this connectivity hypothesis; a
training-only gain would not. Initial affinity should not become a hard
finite-step direction gate without additional evidence. This is a proposed
controlled comparison, with no implementation or training performed.
