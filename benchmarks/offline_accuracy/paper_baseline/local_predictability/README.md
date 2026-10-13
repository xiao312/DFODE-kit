# Local correction predictability diagnostic

## Responsibility and non-goals

Measure local variation of the fixed-asinh chemistry target and the correction
target on the same neighbor graph. This is a diagnostic, not a promoted model,
capacity bound, intrinsic dimension estimate or independent test. No neural fit,
new reference, altered split, target-based neighbor choice or production change.

## Inputs and fixed plan

Read `research/experiments/EXP-20261013-local-predictability.json` before execution.
Use the pinned continued bases and existing prepared arrays for both seeds.
Libraries are the original 10k and the verified disjoint 50k S selection. Queries
are 256 training and 512 fixed U rows per training snapshot, sampled uniformly
from source-ID-sorted rows with RNG 20261013, plus all 1023 D states. Queries exclude
their own source IDs from each library. U/D never enter the libraries.
Source IDs are local to each dataset. T/U share the training namespace; D has a
separate namespace even when its numeric IDs match training IDs. Exclusion uses
the namespace and ID pair, not the integer alone.

Search two fixed spaces: the first 61 standardized state features, and all 119
current correction features. Use Euclidean distance on the stored FP32 features,
computed by direct FP64 differences. Do not refit scales. Rank by distance, with
library order as tie-break. Compare nearest 1 and uniform nearest 8 averages.

For each same graph, compare `z=asinh(d/1e-14)` and `c=z-asinh(b/1e-14)`.
Transfer a neighbor's z, or transfer c around the query base b. Score both with
the unchanged physical rule. Also report coordinate variation, distance, reference
sign differences, snapshot mix and shuffled-library controls. These transfers
use training labels only; query labels are read by the trusted scorer afterward.
They are diagnostic calculations, not a deployed label-lookup prediction rule.

## Interfaces, security and dependencies

`run --project <root> --output <new-dir>` checks provenance; add `--execute` to run.
Inputs are the existing relative paths in protected `research/bindings.json` and
the S report. Outputs are frozen query/library indices, neighbor arrays, physical
transfer arrays, scores, environment and hashes under a new ignored run directory.

`run -> research_contract identity/sandbox + hidden_features resource monitor`;
`neighbors -> NumPy/PyTorch`; `analysis -> pinned reference scorer + stable decoder`.
The GPU worker receives only features, source IDs and library sizes. It cannot
read labels, home, network or write inputs/source. One idle GPU at a time; stop
only our worker on contention. The trusted CPU scorer uses one thread.
No other modules depend on this diagnostic. No evaluator source is modified.

## Verification

Run `python -m pytest tests/test_local_predictability.py -q`.
Expected: exact nearest neighbors, stable ties, self-exclusion, disjoint row
selection, label-free worker inputs and stable reconstruction tests pass.
The run checks query/library identities and recomputes sampled neighbor searches
independently on CPU. A separate review recounts all physical pass/repair/damage
counts from saved arrays. No result can promote a candidate.
