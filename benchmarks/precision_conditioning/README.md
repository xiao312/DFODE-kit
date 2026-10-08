# Precision conditioning experiments

## Responsibility

Measure representational error before training chemistry surrogates. This benchmark
uses explicitly known synthetic increments to separate endpoint subtraction,
coordinate encoding, and state reconstruction. It does not measure neural-network
accuracy or establish a chemical reference solution.

## Inputs and outputs

The default deterministic grid contains nonnegative initial mass fractions and signed
increments spanning 1e-32 through 1e-2. Invalid negative endpoints are excluded.
`--atol` and `--rtol` define a diagnostic error budget
`w = atol + rtol * abs(y0)`; these are evaluation weights, not solver tolerances or
a guarantee on global integration error.

Output is JSON with reference inputs, error distributions, magnitude bins, dtype,
and Python/NumPy versions. Results go under the ignored `runs/` directory. A dry run
prints the planned grid size and destination without writing anything.

## Dependencies and boundary

Python and NumPy only. No solver, model, network, credentials, or GPU access.
This is independent of production preprocessing; later training experiments consume
the same metrics but must retain their real-data lineage and label tolerances.

## Example and verification

```bash
python benchmarks/precision_conditioning/audit.py --dry-run
python benchmarks/precision_conditioning/audit.py --output runs/precision-conditioning/audit.json
python benchmarks/precision_conditioning/build_report.py runs/precision-conditioning/audit.json
python -m pytest tests/test_precision_conditioning_audit.py -q
```

Expected: finite metric summaries, signed targets and exact zeros retained, and
separate counts for increments lost in endpoint subtraction and state addition.
Even FP64 cannot recover an increment that was lost when its endpoint was stored.

The signed-power baseline is `sign(d) * abs(d)**0.1 / 0.1`.
The asinh baseline is `asinh(d/w)`. Both are encoded to FP32 or FP64 and decoded
in FP64. This isolates target quantization; it is not an end-to-end FP32 pipeline.
The transformed-state baseline uses naive endpoint Box-Cox subtraction intentionally
to quantify that failure mode. It is not a stable implementation recommendation.
