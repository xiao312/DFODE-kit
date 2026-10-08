# Flame review content

## Responsibility and non-goals

Present the reviewed flame experiments in a step-by-step HTML report. This
directory contains authored content only, not a fork of the Data app runtime.
It does not run experiments, fetch private artifacts, or select models.

## Interface and configuration

Input: the sanitized snapshot from `../review_snapshot.py`, with `models`,
`reference`, and `cfd` queries, plus optional `runtime_parity`, `flame_profile`,
`historical`, `datasets`, `scaling`, `expanded_reference`, `filter_audit`, and
`heldout` evidence.
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
