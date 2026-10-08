# Flame review content

## Responsibility and non-goals

Present the reviewed flame experiments in a step-by-step HTML report. This
directory contains authored content only, not a fork of the Data app runtime.
It does not run experiments, fetch private artifacts, or select models.

## Interface and configuration

Input: the sanitized snapshot from `../review_snapshot.py`, with `models`,
`reference`, and `cfd` queries, plus optional `runtime_parity` and `flame_profile`.
Output: the `ReportContent` React export used by a prepared report app.
Stable report and component IDs must survive updates. Update the narrative when
the experiment status changes; a data-only refresh is not sufficient.

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
