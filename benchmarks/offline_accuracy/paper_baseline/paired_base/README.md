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
