# Matched-update data-size comparison

## Responsibility and non-goals

Separate additional unique rows from additional optimizer work. Compare the
verified nested 50k and 200k training selections, not a new dataset. No chemistry
generation, test-set access, model selection, CFD deployment, or 1M launch.
This is a controlled Fuel-derived recipe, not the original epoch schedule.

## Frozen protocol

Eight fits: 50k/200k rows x fuel-state/fuel-power x seeds 20261011/20261012.
Every fit uses 6,000 updates, exactly 10,000 rows per batch, and 60M presentations.
All pools are divisible by batch size. Shuffle without replacement within each
epoch. Record every full-pool pass; no partial batch is dropped. The same seed
gives identical initial weights across sizes and representations.

Keep 61 inputs, four 800-unit GELU hidden layers, 58 non-AR outputs, FP32, Adam,
normalized L1, TF32 disabled, and stable FP64 reconstruction. State learning
rates are 1e-3 for updates 1–2000, 1e-4 for 2001–4000, and 1e-5 for 4001–6000;
reset Adam at each transition. Power rates are 1e-3/1e-4/1e-5/1e-6 in four
1,500-update blocks; retain Adam moments. These boundaries are fixed before fits.
Schedules differ between recipes, but are identical between sizes within each
recipe. Do not attribute a between-recipe difference to coordinates alone.

Fit preprocessing on each seed's original selected 50k rows, independently for
each recipe. Freeze those exact arrays for BOTH sizes. Rebuild and hash-check
them at verification. No development data enter preprocessing or fitting. Keep
all 1,023 development states and both error-scale grids unchanged. Evaluate the
entire training pool at fixed 1,000-update intervals, plus final replay and
independent acceptance/physical checks. Retain the final checkpoint, not the best
development checkpoint. Diagnostic cost differs with pool size and is reported;
equal updates/presentations do not guarantee identical elapsed time.

## Interface and outputs

`python -m benchmarks.offline_accuracy.paper_baseline.matched_work.run
--campaign <completed-200k-campaign> --previous <completed-50k-campaign>
--original <original-10k-dataset> --base-root <original-paired-campaign>
--recipe fuel-state --seed 20261011 --training-count 50000
--output <new-directory>` is a read-only plan. Add `--execute` to fit.

`python -m benchmarks.offline_accuracy.paper_baseline.matched_work.campaign`
accepts the same four source paths and a new `--output` root. It previews eight
commands; `--execute` runs them sequentially. Require explicit idle GPU selection
and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Each fit has a 1,200-second internal limit
and a 1,500-second outer limit. Failure stops the campaign. There is no CPU fallback.
All output goes to new ignored project run directories; existing artifacts and
shared environments stay unchanged. Use allocated compute nodes, not HPC login
nodes. Credentials and private source assets never enter Git or public reports.

Output: runtime/config/source hashes, weights, optimizer, preprocessing, row IDs,
fixed-update histories, predictions, full metrics, and result-bound verification.
Campaign verification checks paired initial weights and preprocessing, exact work
counts, nested row IDs, development predictions' row IDs, and all eight results.

`python -m benchmarks.offline_accuracy.paper_baseline.matched_work.review
<existing-snapshot.json> <completed-campaign> --output <new-snapshot.json>`
adds sanitized summaries to the existing review. Download only each fit's
result, verification and environment JSON, plus campaign verification JSON.
The compiler requires all eight result hashes to match the remote campaign
verification. It checks paired configuration, work, preprocessing and source
identities again. It does not replay weights locally or expose private paths.
Output queries are `fuel_matched_models`, `fuel_matched_history` and
`fuel_matched_pairs`. The latter reports 200k minus 50k acceptance in percentage
points for each recipe, seed and policy; it does not select the best fit.

## Fixed-data budget extension

`budget_campaign` accepts the same source paths plus `--baseline <verified-matched-work-root>`.
It previews four fresh 200k fits; `--execute` runs both recipes and both seeds.
Each receives 18,000 updates, batch 10k and 180M presentations. Compare against
the checked 6,000-update 200k fits, not the source fixed-epoch runs. Stretch each
recipe's learning-rate stages threefold; retain its Adam reset policy. This is
a budget-and-proportional-schedule comparison, not low-rate continuation.
The first 6k batch selections, initial weights, 50k scalers, full training IDs,
development IDs and both error policies must match. Diagnostics remain every
1,000 updates. There is no checkpoint selection, test access or data expansion.
The per-fit limits remain 1,200 seconds for fitting and 1,500 seconds overall.

The single-fit `run` accepts `--budget-baseline <verified-6k-fit>` only with
200k rows. Without it, the original 6k protocol is unchanged. Baseline artifacts
are checksum-checked before fitting; array identities, preprocessing and initial
weights are checked again before the result is accepted. `budget_campaign`
binds all four new results and the original eight-result campaign verification.
`budget_review` emits sanitized `fuel_budget_models` and `fuel_budget_history`
queries; it never replaces matched-work or earlier source-recipe evidence.
Use `python -m benchmarks.offline_accuracy.paper_baseline.matched_work.budget_review
<current-snapshot> <completed-budget-campaign> <verified-matched-work-root>
--output <new-snapshot>`. It also emits `fuel_budget_pairs`, retaining both seeds
and policies. Missing results, altered work or baseline identities fail closed.

## Dependencies and verification

`campaign -> run -> fit/plan`; `run -> paper_baseline.data/coordinates/fit/run`
reuses the checked data gate, reconstruction and independent physical evaluator.
No production module depends on this experiment. Verified summaries feed the
existing review; report code cannot affect fitting or evaluation.
`review -> plan / paper_baseline.review.primary -> sanitized queries` has no
Torch dependency and preserves every unrelated query and the report ID.
`budget_campaign -> run / budget_checks` reuses the fitting and validation boundary.
`budget_review -> review / plan` supplies the fixed-data review, without training.

`python -m pytest tests/test_matched_work.py tests/test_matched_review.py
tests/test_budget_work.py tests/test_paper_baseline.py -q`

Expected: exact schedule/reset boundaries, equal presentations, unchanged pools,
common normalization, GPU reload parity, refusal of invalid saved identities,
complete pairs, and correct percentage-point differences in review output.
Local CPU checks may skip the GPU test; execute that test on the server GPU.
