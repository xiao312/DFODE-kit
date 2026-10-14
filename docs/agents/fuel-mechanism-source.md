# Fuel study: mechanism source check

Checked on 2026-10-08. This note supports
[checkpoint 2](https://github.com/xiao312/DFODE-kit/issues/3).

## Confirmed mechanism

The Fuel study names the **Okafor et al. (2018)** mechanism. It reports
**59 species and 356 elementary reactions**. Reference 34 identifies the
source paper. Thus, the identification does not depend only on these counts.
[Study, introduction and reference 34](https://arxiv.org/html/2507.08277v2)

The source authors are E. C. Okafor, Y. Naito, S. Colson, A. Ichikawa,
T. Kudo, A. Hayakawa, and H. Kobayashi. The paper is in *Combustion and Flame*,
volume 187, pages 185-198 (2018). Its DOI is
[10.1016/j.combustflame.2017.09.002](https://doi.org/10.1016/j.combustflame.2017.09.002).

## Study conditions

The study uses these conditions for its training-state source:

| Quantity | Value |
| --- | --- |
| Configuration | 1D freely propagating premixed laminar flame |
| Fuel blend | 60% NH3 / 40% CH4 |
| Oxidizer | Air |
| Equivalence ratio | 1 |
| Unburnt temperature | 300 K |
| Pressure | 1 atm |
| Mesh | 500 cells; 10 points across the flame thickness |
| Sampling interval | 1e-6 s |
| Simulation duration | 2.5e-3 s |
| Initial state count | Approximately 1,250,000 |
| Label interval | 1e-6 s |
| Label solver | Cantera / SUNDIALS CVODE |

The percentage basis is not explicit in the inspected methods text. Confirm
it from the original case before a claim of exact reproduction.
[Study, sections 2.1 and 2.2](https://arxiv.org/html/2507.08277v2)

## File status and acceptance gate

The mechanism identity is confirmed. The exact kinetics, thermodynamic, and
transport files are **not yet verified**. The publisher page was found, but
full-page access returned HTTP 403. A verified official download URL and
checksum were not obtained. This is an access limit, not proof that the files
are unavailable.

Before the NH3/CH4 case can run:

1. Obtain the original study file or the original mechanism supplement.
2. Record its source, license, file names, and SHA-256 checksums.
3. Record any Cantera conversion command and converter version.
4. Check species order, species count, reaction count, and thermodynamic data.
5. Compare rates at recorded test states if a second file format is available.

Matching counts alone does not close this gate. Keep the 2018 detailed model
separate from later reduced or modified Okafor models.

## Scope of the first benchmark

Use the installed Cantera H2 and CH4 mechanisms while the NH3/CH4 source is
checked. A homogeneous constant-pressure, adiabatic benchmark is a new test
design, not a reproduction of the paper's flame case. Record that distinction
in each report. Add the flame case as a separate comparison after the file
gate is closed.
