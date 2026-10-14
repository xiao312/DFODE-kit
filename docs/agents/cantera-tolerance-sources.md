# Cantera tolerance examples

Checked on 2026-10-08. Scope: Cantera version `3.2.0`.

## Result

Use the Cantera default as the baseline: `rtol=1e-9`, `atol=1e-15`.
The source defines these values in `ReactorNet`. The separate sensitivity
defaults are `rtol_sensitivity=1e-4` and `atol_sensitivity=1e-6`.
[Versioned source](https://github.com/Cantera/cantera/blob/v3.2.0/include/cantera/zeroD/ReactorNet.h)

The table gives examples from the same release. These are published settings,
not a survey of values used in research.

| Example | State `rtol` | State `atol` | Basis |
| --- | --- | --- | --- |
| Constant-pressure, adiabatic reactor | `1e-9` | `1e-15` | Uses `ReactorNet` without an override; values come from its defaults. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/reactor1.py) |
| Constant-pressure reactor with sensitivity analysis | `1e-6` | `1e-15` | Sets the state tolerances. It separately sets both sensitivity tolerances to `1e-6`. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/sensitivity1.py) |
| Diesel-type engine | `1e-12` | `1e-16` | Sets both state tolerances and a temperature advance limit. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/ic_engine.py) |
| Interactive reaction-path diagram | `1e-12` | `1e-12` | Sets both state tolerances for an ideal-gas, constant-pressure reactor. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/kinetics/interactive_path_diagram.py) |
| Porous-media burner reactor cascade | `1e-4` | `1e-9` | Sets state tolerances for a more complex reactor model. Its sensitivity settings are separate. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/porous_media_burner.py) |

## Limits and exclusions

- The examples do not establish a single common tolerance pair. Their models and
  purposes differ. The Cantera guide says that defaults can require adjustment
  for a specified system. [Reactor guide](https://cantera.org/3.2/userguide/reactor-tutorial.html#setting-tolerances)
- Do not use sensitivity tolerances as state tolerances. Cantera exposes separate
  properties for these equations. [Python API](https://cantera.org/3.2/python/zerodim.html#cantera.ReactorNet)
- The `custom.py` ignition example uses SciPy VODE. It is not evidence for the
  `ReactorNet` CVODES defaults. [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/custom.py)
- The `1D_packed_bed.py` example uses a separate DAE solver. Its explicit
  `rtol=1e-6`, `atol=1e-14` pair is not a `ReactorNet` CVODES setting.
  [Source](https://github.com/Cantera/cantera/blob/v3.2.0/samples/python/reactors/1D_packed_bed.py)
- A read-only search of all Python examples under `samples/python/reactors` and
  `samples/python/kinetics` in tag `v3.2.0` found no explicit `atol=1e-20` or
  `atol=1e-24` setting. This bounded search does not show that these settings are
  unused elsewhere. [Release tree](https://github.com/Cantera/cantera/tree/v3.2.0/samples/python)

## Implication for the proposed ladder

This is a recommendation, not a Cantera requirement. Keep the default pair as a
baseline. Use smaller tolerances as convergence tests. Do not describe `1e-20`
or `1e-24` absolute tolerances as common based on these sources. Do not call the
smallest requested tolerance a verified reference until the results converge.
