# Input-cell reference diagnostic

Test whether a single FP32 neural input can require incompatible GBCT outputs.
This is not a trainer, a new tolerance, or a full-data precision certificate.

Inputs: registered experiment, common training-only states, frozen preprocessing,
bound species and mechanism. Outputs: exact paired states, five-check reference
records, uncertainty-expanded output intervals and collision counts. No labels
are replaced. No development or confirmation data enters this module.

The GBCT decoder also receives FP64 initial composition. Thus compare normalized
network-output intervals, not physical increments alone. Include the saturated
zero-endpoint preimage, reference uncertainty and outward numerical margins.

Dependencies: existing input/GBCT coordinates, reference chemistry, immutable
bindings and prior common-row artifacts. No production dependents. Four bounded
single-thread subprocesses use the pinned environment; no GPU or secrets.

Run `python -m benchmarks.offline_accuracy.paper_baseline.input_cells.run
--project .` for read-only checks, then add `--execute` for a new diagnostic.
Verify with `pytest tests/test_input_cells.py -q`. Expected: exact collision,
mass/positivity, finite encode branches, uncertainty union and saturation tests.
