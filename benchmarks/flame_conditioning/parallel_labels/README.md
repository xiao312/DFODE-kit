# Resumable parallel reference labels

## Responsibility and non-goals

Generate the existing fixed-temperature, fixed-volume CVODE labels faster.
Do not change tolerances, row sampling, physical rejection rules, or test access.
This is CPU reference integration, not CPU neural training. Existing datasets,
source cases, mechanisms and environments are read-only.

## Interface and configuration

`python -m benchmarks.flame_conditioning.parallel_labels.run <source> --config
<dataset.json> --output <new-directory> --workers 4` is read-only discovery.
Add `--execute` to generate labels. Worker count is explicit, from 1 to 8;
each worker has one numerical thread. Use allocated compute resources, not an
HPC login node. The current ceiling is 201,000 candidate training rows, allowing
some exclusions before a 200,000-accepted-row fit. One million is a later gate.

The dataset configuration is unchanged. `--chunk-rows` defaults to 256 (1–2048).
The wall limit includes sampling and labeling. On timeout, the process exits
nonzero and retains committed chunks. Use `--resume-source <partial-directory>`
with a new output directory and the identical configuration. It verifies source,
code/runtime identity, input hashes, chunk hashes, row IDs and non-overlap.
Uncommitted temporary files are ignored. Completed chunks are never regenerated.
Each resume gets a new bounded wall budget and records its parent hash.

`--reuse <complete-dataset>` imports a checked training prefix and the unchanged
development labels. Only training count and wall limit may differ. All input and
lineage arrays must match exactly; source, mechanism, Cantera and chemistry
settings must match. A fresh reference audit is still required before training.
Reuse and resume are mutually exclusive. Final `inputs.npz`, `labels.npz`, and
`manifest.json` retain the existing training/audit interface. Chunk files and
row-level solver records remain available for inspection.

`python -m benchmarks.flame_conditioning.parallel_labels.benchmark <dataset>
--output <new.json>` previews a benchmark. Add `--execute` for identical rows
with 1, 4 and 8 workers, in two repetitions. Each measurement includes worker
startup and transport. Compare every increment and acceptance flag exactly with
the saved serial labels. Report throughput; do not assume linear speedup.
The default is 2048 evenly spaced training rows per run, with a 300-second limit.

## Dependencies, outputs and security

`run -> storage + worker -> chemistry.EndpointIntegrator`; sampling uses the
existing `augmentation.sample_split`. `benchmark -> worker + checked data`.
Dependents are the existing dataset reader, reference audit and GPU trainers.
Only the parent writes chunks and manifests. Each spawned process owns its own
Cantera mechanism and reactor. No network, credentials or production changes.
Output manifests record versions, source revision, implementation hashes,
worker/chunk settings, reuse hashes, row counts, elapsed time and failures.

## Verification

`python -m pytest tests/test_parallel_labels.py tests/test_flame_augmentation.py -q`

Expected: parallel/serial labels agree exactly; changed inputs, corrupt chunks,
overlap and changed resume contracts fail; dry-run writes nothing; partial runs
resume into new directories; excluded and unfinished rows never become zeros.

Example: prepare 200k candidates with four workers, then run the independent
audit before any GPU fit. Larger data from the same snapshots tests sample
density only. Broader source coverage needs a separate split/provenance gate.
