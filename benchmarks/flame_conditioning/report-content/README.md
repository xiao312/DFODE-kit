# Flame review content

## Responsibility and non-goals

Present the reviewed flame experiments in a step-by-step HTML report. This
directory contains authored content only, not a fork of the Data app runtime.
It does not run experiments, fetch private artifacts, or select models.

## Interface and configuration

Input: the sanitized snapshot from `../review_snapshot.py`, with `models`,
`reference`, and `cfd` queries, plus optional `runtime_parity`, `flame_profile`,
`historical`, `datasets`, `scaling`, `expanded_reference`, `filter_audit`, and
`heldout`, `heldout_sampling`, and `heldout_reference` evidence.
Optional `cfd_tolerance` evidence compares two CVODE-only restart runs. Keep
its final-state difference distinct from one-step learned-increment errors.
Output: the `ReportContent` React export used by a prepared report app.
Stable report and component IDs must survive updates. Update the narrative when
the experiment status changes; a data-only refresh is not sufficient.
The reserved-snapshot tables use a pre-test display rule: show the largest
completed matched 4x800/10k-update run, zero baseline, and historical source-formula
controls. Keep uniform and temperature-balanced populations in separate tables.
Retain every frozen model's scores in source data. Never select table rows by
their test performance.
Reference rows distinguish zero numerical increments from unresolved nonzero
increments. A relative-resolution fraction over all species must not be described
as the unresolved-label fraction; inert-species zeros have no relative error.
Model rows expose the training seed. Primary four-target charts retain seed
20261009; the conventional-only 20261010 repeat has its own explanatory block.
Do not merge repeated fits into a best-seed summary.
The compact density table retains every completed primary-seed conventional
4x800/10k-update fit, in ascending actual training count. Missing larger fits do
not appear as zero. It does not select rows by validation or test performance.
Training-only scales are refit at each size; this is a full-protocol data-size
comparison, not an isolated fixed-scaler ablation.
`validation-selection.mjs` selects the largest completed primary-seed four-target
comparison for the main validation chart and species table. It uses training
count, never error values. It requires the fixed FP32/4x800/GELU/L1/10k-update
recipe and rejects missing or duplicate targets. Earlier sizes remain in the
full results and density tables. Verify with
`node --test tests/flame_report_selection.test.mjs`.
Predeclared bin tables show zero, conventional, direct-power, and fixed-hybrid
policies from the largest completed primary run. Temperature bins use the balanced
diagnostic population; magnitude bins use the uniform population. Other models
and both populations remain in source rows. Never select these columns by scores.
`policy-bins.mjs` preserves empty-bin nulls and rejects duplicate or unequal-count
comparisons. Verify with `node --test tests/flame_report_bins.test.mjs`.
Historical raw inverse-domain violations are separate from performed corrections.
Show absent diagnostics as not recorded, not zero. Preserve small nonzero rates
in percentage formatting. Test-batch diagnostics do not become population rates.
The preselected species view shows NH3, CH4, NO, and OH, using each species'
physical-budget p99. This selection is chemical-role based, not chosen from test
errors. The offline balanced-population view is not a domain-average emissions
or flame-speed result. Missing species metrics remain null, never zero.

## Dependencies and dependents

Use the installed Data report skill to prepare the shared runtime. Import its
public components through `../../data-app-public.jsx` after placing this file in
the prepared app's `src/content/report/` directory. Keep the shared runtime's
source actions, table controls, and chart exports. GitHub Pages consumes only the
verified compiled HTML, sanitized data sidecar, and build manifest.

## Security and verification

No credentials, absolute machine paths, model weights, raw mechanism, or session
history belongs in this content or its snapshot. The public report can include
the reviewed temperature profile, which contains positions and temperatures only.

Minimal example: prepare a report app with the reviewed snapshot, place
`ReportContent.jsx` in its content directory, then run the installed
`data-app.mjs build --project-dir <app> --separate-data` command. Expected: authored
content verification passes and `dist/` contains HTML, the data sidecar, and its
build manifest. Read the copied app's AGENTS.md before authoring or building.
Check the rendered report when a browser is available. A build alone is not a
visual or scientific validation. Record any unavailable check honestly.
When publishing through Git, preserve generated HTML/JSON bytes with scoped
`-text` attributes. Git newline normalization can otherwise invalidate the
content-addressed data hash. Fetch the published HTML and sidecar and compare
their SHA256 values with `data-app-build.json`; HTTP 200 alone is insufficient.
