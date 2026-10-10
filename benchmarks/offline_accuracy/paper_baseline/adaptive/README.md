# Residual-calibrated local correction

## Responsibility and non-goals

Test whether a train-calibrated, magnitude-dependent residual coordinate improves
the frozen GBCT/increment baseline. Keep the same 200k rows, two seeds, development
states and error rules. This is an experimental correction, not a new reference
solver, a full MSNN reproduction, uncertainty certification or CFD deployment.
Historical experiments, environments and independent tests remain read-only.

## Frozen experiment

Six GPU fits: `continue`, `local`, `calibrated` for each seed. `plan.py` pins the
two base result hashes and their complete 24-fit campaign verification. Each arm
receives 6k additional updates, batch 10k, Adam reset at the start, cosine learning
rate 1e-4 to 1e-6, physical mean log1p(abs(error)/(1e-15+.1*abs(reference delta))).
Keep the final checkpoint and diagnose every 1k updates. No result-based changes.

`continue` warm-starts the existing 4x800 GBCT model. Both residual arms freeze
that model, append a zero-output-initialized 3x256 GELU network, and decode in FP64:

`prediction = base + (1e-14 + abs(base)) * alpha(base) * correction`.

`local` uses alpha=1. `calibrated` fits alpha from the first 50k training rows:
compute abs(reference-base)/(1e-14+abs(base)); take the 75th percentile per species
in eight fixed bins of log10(1+abs(base)/1e-14), with edges 0,2,...,16. Fewer than
128 rows uses that species' global 75th percentile. Bound alpha to [.01,100],
record raw values and counts, and interpolate log(alpha) between bin centers.
Use constant endpoint extrapolation. Calibration never reads development labels.
This conditional robust scale is our adaptation, not a quoted literature recipe.
It controls representation amplitude, not acceptance tolerances or uncertainty.

Both correction arms use identical normalized physical inputs plus compressed
base outputs asinh(base/1e-14). Fit the added feature mean/sample standard deviation
on the same 50k prefix. Seeds, batches, correction architecture, zero initialization
and loss are paired. No true residual enters inference. No clipping or positivity
projection hides failures; record negative endpoints and inherited base inverse
corrections. Exact zero correction must preserve the base, including trace values.

Milestone: at least +5 percentage points primary development acceptance in BOTH
seeds, lower error/allowance p99, and no increase in negative endpoint components.
Also beat the local correction to claim a scale-specific benefit. Compare with
continuation cost; equal updates do not imply equal parameters or elapsed time.
This is a target, not a guarantee. Retain both policy grids, whole-state scores,
physical checks and pass-to-fail/fail-to-pass counts. Repeated development use
limits generalization claims. No independent-test opening or dataset expansion.

## Interface, outputs and security

`python -m benchmarks.offline_accuracy.paper_baseline.adaptive.campaign
--campaign <200k-source-campaign> --previous <50k-source-campaign>
--original <original-dataset> --base-root <original-paired-campaign>
--baseline <verified-matched-target-campaign> --output <new-directory>` previews
the finite queue. Add `--execute` to run. The single-fit `run` also needs `--arm`
and `--seed`; without `--execute` it validates inputs without creating outputs.
Existing outputs are refused. Select one idle GPU and deterministic CUDA workspace.
Use the pinned project GPU environment. No CPU fallback or shared environment edits.
Fit limit 1200 seconds; outer limit 1500 seconds. The queue stops on any failure.

`adaptive.launch` accepts the same path options and `--execute`. It waits up to
six hours for a GPU with no compute processes, memory use below 256 MiB and
utilization below 10%, in two checks five seconds apart. Poll once per minute.
It then runs the focused GPU tests and a read-only real-data gate before the
campaign. Write launcher status beside, not inside, the campaign output.
Without `--execute`, print a read-only GPU discovery and queue plan. Each fit
also checks that its selected GPU has no other compute process before it starts.
This cooperative check is not a scheduler reservation; other users can start
work later. No other user's process is stopped. Mixed-load timing must not be
treated as dedicated-GPU timing.

Save frozen base weights with each checkpoint, all preprocessing, row identities,
environment/source versions, base/result hashes, predictions, training history,
FP64 residual round-trip diagnostics and verification. Store large/private outputs
only in ignored runs; never publish mechanisms, machine paths or credentials.
Timing includes correction setup/fitting; total deployment cost includes the base.

## Dependencies and verification

`launch -> campaign -> run -> fit -> model/coordinates/plan`.
`launch` and `campaign` also read GPU availability through `gpu`; it has no
dependencies on trainers or launchers. Read data through the existing
`matched_work.run.inputs` gate. Load the pinned baseline with checked artifact
hashes; reuse matched-target inference, paired runtime/metrics and the explicit
adapters in `paper_baseline.run.evaluate_and_verify`. Production paths do not
depend on this module. Reviews consume sanitized verified summaries only.

`python -m pytest tests/test_adaptive_representation.py -q -p no:cacheprovider`

Expected: finite bounded scales, sparse-bin fallback, train-only calibration,
continuous interpolation, exact zero-correction identity, finite gradients,
fixed base hashes and exact GPU checkpoint replay. All six live fits must pass
independent acceptance and physical checks before the campaign is complete.
