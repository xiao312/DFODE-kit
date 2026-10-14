# Correction decoder and loss diagnostic

## Responsibility and non-goals

Isolate loss pressure from correction-coordinate conditioning around the pinned
GBCT/increment baseline. Preserve the older adaptive campaign and all datasets.
No CFD, new labels, independent-test access, positivity projection or relaxed
acceptance rule. This is not evidence that a transform guarantees learnability.

## Frozen matrix and success criteria

Eight fresh fits: `linear-log`, `linear-huber`, `asinh-log`, `asinh-huber`, each
with seeds 20261011/20261012. Inspect the loss-only pair first. Use the same 200k
training states, 50k normalization prefix, 1023 development states, frozen base,
119 features, zero-output 3x256 GELU correction, batches and initialization.
6k updates, batch 10k, fresh Adam, cosine 1e-4 to 1e-6, final checkpoint. FP32
network and FP64 reconstruction/loss. No output-coordinate normalization: its
offset is zero and divisor is one. Alpha=1 throughout. No gradient clipping;
record every update's raw global L2 gradient norm, zero clipping count and
nonfinite failures. Nonfinite loss, gradient or prediction stops the campaign.

`linear`: prediction = base + (1e-14 + abs(base))*correction.

`asinh`: prediction = 1e-14*sinh(asinh(base/1e-14)+correction).
Use the incremental hyperbolic identity for abs(correction)<=0.5 so zero exactly
preserves the base AND has a nonzero derivative. Use direct shifted sinh outside
that interval to avoid cancellation when correcting a wrong magnitude. Evaluate
only the selected branch, with no hidden coordinate clipping.

Both losses use u=(prediction-reference)/(1e-15+0.1*abs(reference)). `log` is
mean(log1p(abs(u))); `huber` is mean(0.5*u^2 if abs(u)<=1 else abs(u)-0.5).
The same learning rate is a controlled diagnostic, not each loss's optimum.
Huber's bounded derivative in u does not bound parameter gradients.

Keep primary increment acceptance and whole-state acceptance. Also record
conditional overprediction counts P(abs(pred)>=eta | abs(reference)<=epsilon),
epsilon=[1e-15,1e-12,1e-9], eta=[1e-12,1e-8,1e-6], eta>epsilon; sign errors above
1e-15/(1-.1); normalized p99/max by species and fixed target-magnitude bins;
pass-to-fail/fail-to-pass counts, negative endpoints and cost. Primary rare-event
pair: epsilon=1e-12, eta=1e-8. Target: tenfold lower frequency versus frozen base
in BOTH seeds, lower conditional p99 and no decrease in acceptance on nontrivial
increments (abs(reference)>1e-15/(1-.1)). Report exact denominators; zero baseline
events cannot demonstrate a tenfold gain. A label-resolution-qualified version
is limited to the existing independently audited development subset, with
uncertainty<=0.1*allowance; never call all 200k labels independently resolved.
State scores remain secondary compatibility diagnostics, not a selection goal.

## No-training audit

Before updates, compute oracle corrections from each split's reference labels.
Cast to FP32, decode in FP64, measure physical error/allowance, and repeat with
the adjacent FP32 coordinates in each direction. A rounded-oracle failure is
not itself a proven representation impossibility. Record FP64 oracle results,
nearest/neighbor counts, exact accepted coordinate interval widths and local
allowance/decoder-derivative quantiles. Include per-species and magnitude bins.
This uses labels only for diagnostics, never fitted preprocessing or inference.
Every arm must start from the exact base and replay from saved weights.

## Interface and boundaries

`python -m benchmarks.offline_accuracy.paper_baseline.decoder_loss.run
--campaign <200k-campaign> --previous <50k-campaign> --original <original-data>
--base-root <original-paired-root> --baseline <matched-target-campaign>
--output <new-directory> --arm linear-log --seed 20261011` is read-only discovery.
Add `--execute` to audit, fit and independently verify. `--audit-only` with
`--execute` saves decoder checks without training. `campaign` takes the same path
arguments, previews all eight fits, and executes only with `--execute`.
Outputs must be new. Select an idle GPU with CUDA_VISIBLE_DEVICES; set
CUBLAS_WORKSPACE_CONFIG=:4096:8. Use the pinned project-local GPU environment.
No CPU training fallback. Fits have a 1200-second update budget and a 1800-second
outer limit including verification. Before each fit, allow up to 60 seconds for
the selected GPU to be idle; then stop without touching other processes. Record
occupancy samples during training; flag timing if another GPU process appears.

Outputs: base/correction weights, preprocessing, row identities, FP64/FP32
decoder audits, per-update gradients, histories, predictions, tail diagnostics,
runtime/source/data hashes and result-bound verification. Large arrays and
mechanisms stay in ignored runs; no credentials or private paths in publications.

## Dependencies and verification

`campaign -> run -> fit/model/coordinates/diagnostics` with choices in `plan`.
Reuse `matched_work.run.inputs`, pinned `adaptive.plan` baseline identities,
`adaptive.model` construction/features/reload and train-only preprocessing;
new inference supplies only its own decoder. Reuse explicit evaluation adapters
in `paper_baseline.run`, paired runtime/metrics and independent physical checks.
Historical modules do not depend on this experiment. Reviews consume evidence.

`python -m pytest tests/test_decoder_loss.py -q -p no:cacheprovider`

Expected: exact zero identity with correct nonzero gradients, branch continuity,
FP64 and FP32 oracle tests, exact Huber/log formulas, null empty-bin frequencies,
no false resolved-label claims, GPU replay and a complete paired eight-fit matrix.
