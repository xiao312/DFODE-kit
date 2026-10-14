# Historical checkpoint conversion

## Responsibility and non-goals

Convert only the two inspected study checkpoints into non-pickle numerical arrays.
This module does not train, score a test set, run original scripts, or modify the
working CFD or research environments. Historical weights are a comparison control,
not proof of the checkpoint used in a published figure.

## Inputs, outputs and configuration

Input: an exact allowlisted checkpoint hash and a model kind (`state-boxcox` or
`signed-power`). Require the known 61 -> 800 -> 800 -> 800 -> 800 -> 58 structure,
four normalization vectors, finite numbers and positive normalization scales.
Output: numerical NPZ arrays and a manifest recording source/output hashes, shapes,
dtypes, converter version, external mechanism/species contract and restrictions.
Refuse an existing output. A dry run checks the checkpoint identity without loading
pickle. Later consumers must use `numpy.load(..., allow_pickle=False)`.

## Dependencies and dependents

Use a new project-owned Python 3.10 environment with pinned CPU Torch 2.10.0 and
NumPy 2.2.6. Download verified wheels through the controller and install offline;
record every wheel hash. Do not update the existing Torch 2.4.1/2.5.1 environments.
The historical inference adapter consumes the arrays, not the old checkpoint.

## Security boundary

Torch releases through 2.9.1 have a reported `weights_only` loading flaw. The
[official advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
lists 2.10.0 as patched. Restricted loading is not a guarantee against every
malformed file. Use the allowlisted hash, bounded file size, a read-only container
root and source mounts, no network or credentials, memory/CPU/time limits, and only
one new output directory writable. Never fall back to `weights_only=False`.
For the conventional checkpoint, allow only the explicitly reviewed NumPy array
and FP64 dtype classes required for its normalization vectors. Reject unexpected
keys, types, shapes, nonfinite values, and object arrays. Do not publish original
or converted weights without verifying redistribution rights.

## Minimal example and verification

Run `convert.py --kind signed-power --checkpoint <original.pth> --contract <source-manifest.json> --output <new-dir>
--dry-run` in the converter environment. Expected: the known hash and conversion
plan are printed, and no output is written. Omit `--dry-run` only within the
restricted conversion container. Verify the exported file hash and load all arrays
with `allow_pickle=False`; check the expected shapes and source model contract.
