# Flame review content

## Responsibility and non-goals

Present the reviewed flame experiments in a step-by-step HTML report. This
directory contains authored content only, not a fork of the Data app runtime.
It does not run experiments, fetch private artifacts, or select models.

## Interface and configuration

`method-catalogue.mjs` owns canonical representation and recipe display names,
historical ID aliases, and diagram steps. The selection helpers import its names.
Do not rename immutable run IDs or maintain a second display-name dictionary.
The catalogue describes recipes; executable experiment plans still own parameters.
`node scripts/build_research_method_catalogue.mjs --write` generates the linked
GitHub review page. Without `--write`, it checks for drift and writes nothing.
`ReviewGuide.jsx` explains the three splits and shows saved training/development
scores. It does not run a new test. Missing independent-test scores stay unknown.
`node scripts/prepare_research_review.mjs <snapshot.json>` validates the reviewed
method IDs without writing. Add `--apply` to bind split metadata to the guide's
component IDs; it preserves evidence rows, artifact identity and data timestamps.
Training local-table scores include each queried training point in the table;
they are resubstitution scores, not leave-one-out validation.
The tolerance discussion is a proposal, not an amendment to saved pass criteria.
Verify with `node --test tests/method_catalogue.test.mjs` and the generator's
default read-only check. The compiled report and generated Markdown both depend
on this registry; the architecture graph records this shared dependency.

Input: the sanitized snapshot from `../review_snapshot.py`, with `models`,
`reference`, and `cfd` queries, plus optional `runtime_parity`, `flame_profile`,
`historical`, `datasets`, `scaling`, `expanded_reference`, `filter_audit`, and
`heldout`, `heldout_sampling`, and `heldout_reference` evidence.
Optional `cfd_tolerance` evidence compares two CVODE-only restart runs. Keep
its final-state difference distinct from one-step learned-increment errors.
Output: the `ReportContent` React export used by a prepared report app.
The optional `offline_*` queries from `offline_accuracy.review` add the current
representation experiment through `OfflineAccuracy.jsx`. Controls select a fixed
seed, size, absolute floor, and component or whole-state criterion. Both seeds
and all models remain in the table. CPU cost uses process seconds. Prior CFD
evidence stays below, but is not the current experiment's decision gate.
Stable report and component IDs must survive updates. Update the narrative when
the experiment status changes; a data-only refresh is not sufficient.
Optional `refinement_*` queries from `offline_accuracy.refinement.review` add
the later controlled implementations through `Refinement.jsx`. Preserve the
original offline queries. Keep both seeds, failed/unknown distinctions, physical
loss controls, the frozen base cost and the fixed acceptance grid explicit.
No result-based model selection is permitted in the comparison controls.
Optional `improve_*` queries add the acceptance-directed adaptation campaign in
`Improvement.jsx`. Keep the seven first-stage methods, three later input candidates
and two separately labelled non-learned kinetics controls. The latter use actual
mechanism evaluation at inference and cannot be counted as neural model gains.
Keep both seeds, including failed trials as unknown scores. Compare with each
seed's 4,000-update conventional base.
Show gain and damage separately, and distinguish local table query cost from
neural inference. These are development results, not independent test selection.
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
Label negative endpoint rates as species-component fractions, not cell fractions.
Keep budget, heat-error, negative-component, and inverse-correction definitions
with each new-model, historical-control, and reserved-test source query.
When `reference_recovery` is present, show the strict rejection and amended
eligibility counts before the test scores. State that raw signed labels remain
unchanged and the amendment occurred after reference inspection but before model
scoring. Never describe the amended test as the unchanged original protocol.
Optional `input_support` evidence is a post-score diagnosis. Keep validation,
uniform-test, and balanced-test rows separate. Distinguish raw physical ranges
from transformed-feature scales. A range mismatch is not a causal ablation or
permission to tune against the same test. Retain all feature rows in source data.
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
