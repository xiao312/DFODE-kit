# Training-derived species distance diagnostic

## Responsibility and non-goals

Test whether one fixed distance for all species hides useful local correction
structure. Compare a training-derived diagonal distance with the existing 119-input
Euclidean control. This is a diagnostic, not a promoted predictor. It does not
change labels, tolerances, libraries, query rows, bases or production code.

## Inputs, rule and outputs

Read `research/experiments/EXP-20261013-species-distance.json`. Reuse the hash-bound
local-predictability artifacts for both seeds. Exclude all 1,024 training query
IDs from metric calibration. Use the remaining 48,976 library rows for both the
10k and 50k search, and disclose this extra training-label exposure at the 10k rung.
No U or D labels enter calibration. Each species has a fixed metric at inference.

For each feature j and correction coordinate c_i, calculate Spearman correlation
rho_ij on calibration rows (average ranks for ties). Set
`w_ij = 0.1 + 0.9 * 119 * rho_ij^2 / sum_j(rho_ij^2)`.
If the sum is zero, use all-one weights. Search with Euclidean distance on
`sqrt(w_i) * features`, in FP64. Mean weight is one and every weight is at least
0.1. These constants are fixed before scoring. This simple rule detects marginal
monotone association, not conditional relevance or a complete nonlinear metric.

Return top eight neighbors per query and species, with stable library-order ties
and namespace-qualified self exclusion. Compare nearest-one and uniform-eight
transfers of correction coordinates; full-target transfer is a secondary check.
Use the same graph for both transfers. Recompute the global control and require
exact neighbor-index parity with its saved graph. Output weights, calibration IDs,
graphs, physical predictions, scores, independent recounts and artifact hashes.

## Security and dependencies

`run -> local_predictability artifacts + research_contract admission/sandbox`;
`metric -> NumPy/SciPy ranks`; `worker -> local_predictability.neighbors/PyTorch`;
`analysis -> stable decoder + pinned physical scorer`;
`local_predictability.review -> saved physical arrays`.
No production module depends on this module. The trusted controller fits weights
from training labels. The isolated GPU worker receives only features, IDs and
frozen weight arrays. It has no query labels, home or network. One idle GPU is used;
the existing monitor stops only our worker if another workload appears. CPU
calibration and independent verification use one thread.

## Interface and verification

Read-only example: `python -m benchmarks.offline_accuracy.paper_baseline.species_distance.run --project <root> --output <new-run>`.
Add `--execute` only after the provenance, resource and isolation checks pass.
Run `python -m pytest tests/test_species_distance.py -q`.
Expected: training-query exclusion, rank ties, weight floor, constant-target
fallback, species-specific gathering, physical reconstruction and neighbor parity
tests pass. Verify three fixed query rows per species/library on CPU. Independently
recount every score with `local_predictability.review`. No diagnostic can promote
a candidate or open confirmation.
