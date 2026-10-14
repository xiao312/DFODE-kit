# Initial-rate features and Euler increment: an optional diagnostic

Status: source check, not a selected method or a verified increment predictor.
Scope remains the pinned 59-species mechanism, fixed T and V, h=1e-6 seconds;
the historical score excludes AR, but kinetics must retain all 59 species.

## Formula, units and reactor assumptions

For a closed homogeneous gas with no surface sources and fixed volume, Cantera's
species equation reduces to m*dY_i/dt=V*W_i*omega_i. With rho=m/V this gives
f_i(T,rho,Y)=W_i*omega_i/rho. The explicit initial-rate estimate is d_Euler=h*f_i.
This is the closed-reactor specialization, not a formula for open reactors,
surface chemistry or transport.
[Cantera 3.2 ideal-gas reactor equations](https://cantera.org/3.2/reference/reactors/ideal-gas-reactor.html)

For a bulk gas, `net_production_rates` is in kmol/m^3/s, not reaction rates of
progress and not kg/m^3/s. Surface rates instead have an area denominator.
[Cantera 3.2 kinetics API](https://cantera.org/3.2/python/kinetics.html#cantera.Kinetics.net_production_rates)
`molecular_weights` is kg/kmol; `density_mass` is kg/m^3. Thus f has units 1/s
and h*f is a dimensionless mass-fraction increment. Prefer `density_mass` for
explicit units: `density` and `TD` depend on the phase's mass/molar basis.
[Cantera 3.2 thermodynamic API](https://cantera.org/3.2/python/thermo.html)

Closed-system mass conservation and fixed V keep rho fixed during the step.
Set the initial state from its T, P and full Y, then retain that initial rho.
Pressure generally changes with composition; repeatedly imposing initial P
would change the physical problem. Temperature is held fixed by disabling the
energy equation; this is not an adiabatic temperature evolution.
[Cantera reactor configuration](https://cantera.org/3.2/python/zerodim.html#cantera.Reactor.energy_enabled)

## Existing implementation and input checks

Do not duplicate the existing [rate controls](../../benchmarks/offline_accuracy/improve/physics_prior.py):
they already implement `rate-euler` and `frozen-exponential`. Inspect their saved
artifact identities and populations before deciding whether a new diagnostic is
needed; source availability alone does not establish a completed compatible run.
The [current chemistry adapter](../../benchmarks/flame_conditioning/chemistry.py)
validates nonnegative FP64 Y with sum error <=1e-10, sets unnormalized fractions
then TP, and uses fixed rho in its direct-increment RHS. Reuse those semantics.

Cantera provides `set_unnormalized_mass_fractions` specifically to avoid forcing
sum(Y)=1. Do not silently substitute normalization or clipping: either can change
small rates. Explicitly set mass basis if using TD, and verify species names,
mechanism hash, state order, finiteness, density positivity and stored-state replay.
[Cantera composition setter](https://cantera.org/3.2/python/thermo.html#cantera.ThermoPhase.set_unnormalized_mass_fractions)

Compute physical rates from declared physical inputs, never standardized network
features mistaken for T/P/Y. Record whether inputs are original FP64 or a declared
FP32 round trip. If RHS features use more precise inputs than the control, that is
an experimental factor, not a free representation gain. Fit any feature scaling
on training data only; scaling never changes the physical acceptance allowance.

## Why a rate is not a certified increment: mathematical deductions

The desired increment is the integral of f along the evolving state, not h*f at
the initial state. For a smooth autonomous RHS, its Taylor expansion is
d=h*f(Y0)+(h^2/2)*J_f(Y0)*f(Y0)+O(h^3), at fixed T and rho. Large or rapidly
changing kinetics can make the omitted term important even when h looks small.
The norm of f alone is not a stiffness or accuracy certificate.

For the illustrative decay y'=-k*y, exact d=y0*(exp(-k*h)-1), but Euler gives
-k*h*y0. Euler's endpoint is negative for k*h>1 and linearly unstable for k*h>2.
This calculation is not a claim about any sampled species. Coupled reversible
chemistry can also change rate direction during the interval, so an initial rate
sign need not equal a finite-step increment sign. Near cancellation, a small net
rate can be sensitive to state precision and rate arithmetic. FP64 evaluation,
mass conservation or one small RHS value alone does not certify target accuracy.

## If the family is selected

First score the unmodified Euler control on a fixed training-only population,
with existing references, exact signed increments and the unchanged tolerance.
Report species/regime acceptance, nontrivial signs, tails and negative endpoints;
retain failures without clipping or replacing historical labels. Verify rates
against the existing RHS at zero increment and check conservation residuals.
These are implementation checks, not an independent reference integration.

RHS features are deterministic functions of the permitted state and mechanism.
They expose known chemical structure and incur kinetic computation; they do not
add independent physical information or query-reference access. A learned-feature
trial would need a matched no-RHS control and declared total input exposure.

Measure full live inference: state setup, kinetic-rate evaluation, conversions,
feature scaling, transfers and any network/decoder work. Cached-rate throughput
is not deployable end-to-end latency. Separate mechanism initialization from warm
calls, retain actual CPU/GPU settings and compare with the anchor under the same
batch policy. Do not assume one RHS call meets the 1.5x anchor latency gate.
No extra scientific paper is needed for this limited formula and diagnostic;
the error examples above are deductions, not evidence of chemistry effectiveness.
