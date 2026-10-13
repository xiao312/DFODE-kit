# Paired frozen-base error diagnostic

Responsibility: measure repeatable versus seed-dependent prediction errors.
This module does not train, select weights, deploy a reference oracle or qualify
a candidate. Both bases already saw the training pool.

Inputs: registered JSON plan, pinned prepared base arrays, source IDs and fixed
U selections. Output: two deterministic mean predictions, protected scorecards,
independent count checks and grouped error agreement. Exclude the union of both
seeds' U and historical head-fit IDs. Align and assert all physical row fields.

Dependencies: research-contract adapter/replay, kinetic-features independent
recount, frozen prepared caches. No production dependent. Trusted analysis has
labels, but mean construction accepts base predictions only. This is cached
arithmetic, not a claim of isolated fresh inference or measured deployment cost.
No network, secret, fitting or confirmation input is required.

Example: `python -m benchmarks.offline_accuracy.paper_baseline.paired_base.run
--project .` checks the plan; add `--execute` to save a new result. Never overwrite.
Run with one BLAS/OMP thread. Verify:
`python -m pytest tests/test_paired_base.py -q`.
Expected: alignment rejects duplicates/mismatches, mean has no label dependency,
zero/constant correlation is explicit, and pass-overlap identities hold.

The separately registered `bounded.py` diagnostic uses the saved common rows.
It applies a declared cap of 0.1 times an anchor-prediction scale to the physical
midpoint adjustment. This cap does not enter the physical scorer. It stays on
the convex segment and thus preserves nonnegative endpoints when both input
endpoints are nonnegative. `python -m
benchmarks.offline_accuracy.paper_baseline.paired_base.bounded --project .`
checks the plan; `--execute` saves results. No fitted parameters or new labels.

`screen_prepare`, `screen_worker`, and `screen_review` perform the separately
registered full-population test. `gradient_probe.run --experiment
EXP-20261013-bounded-base-screen --project . --output <new-dir> --execute`
uses the existing resource lease/admission and Bubblewrap guard. Trusted staging
holds references outside worker mounts. Each worker sees only one population's
states/IDs and two pinned base model/preprocessing directories. It saves raw
base predictions, bounded predictions, exact reload evidence and full-pipeline
timing. The independent review uses the protected scorer and unchanged gates.
Review with `python -m
benchmarks.offline_accuracy.paper_baseline.paired_base.screen_review --project .
--run <run>`. No automatic incumbent update or confirmation access.

`shrink.py --project . --execute` (as a module) is a separate registered cached
diagnostic. It restricts the bounded output to the interval between anchor and
zero. Magnitude cannot increase; sign cannot reverse. It cannot repair wrong
base signs, and is not a positivity fix or a new accuracy allowance. Without
`--execute`, only validate the plan and parent evidence. No new inference claim.

`EXP-20261013-strict-consensus-screen` changes that restriction to the interval
between b and0.9b. This retains nonzero base signs, unlike the zero-inclusive
restriction. The same state-only worker and full review support the explicit
`mode: strict-shrink`; historical `bounded` behavior is unchanged. This still
cannot repair severe or wrong-sign base errors. It is not a universal solution.

`gpu_model` is a separately registered execution variant. It keeps the two FP32
networks and CPU input transformation, shares input preparation, and reconstructs
GBCT outputs on GPU in FP64. It preserves the historical inverse-domain rule and
returns its mask. It does not fit or change weights. CUDA and NumPy transcendental
functions need not be bit-identical: retain both outputs and quantify differences.
`screen_worker` selects this only when `backend: gpu-decode` is registered. Both
full physical scoring and arithmetic parity are required; no silent fallback.
The latency record includes the old anchor and an optimized single-base control.
Verify `pytest tests/test_consensus_gpu.py tests/test_paired_base.py -q` in the
pinned environment. Expected: inverse branches, masks, finite results, inactive
species and scalar agreement. Full replay remains a server-only evidence check.
The reviewed GPU decoder rejects invalid states and encoded overflow before its
branch substitutions. The validation revision is replayed in a new run directory;
old outputs remain unchanged. This check adds a device synchronization, so timing
is measured again. No training is repeated.
