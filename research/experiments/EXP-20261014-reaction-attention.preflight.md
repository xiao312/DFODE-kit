# Reaction-aware model: implementation complete, GPU admission blocked

## Authorized repair and implementation recovery (2026-10-14)

The owner authorized the narrow GPU UUID repair and integrity-lock update. Source
`c1caf83` passes32 focused tests and the full isolation probe on GPU2, with exactly
one visible device. The host query confirms index2 is device minor2. Scoring,
mounts, labels, network isolation and the scientific manifest remain unchanged.

The registered infrastructure retry then stopped before any optimizer step:
the common-row reference payload has no optional cached `features` column.
The worker incorrectly indexed it after already computing state-only features.
The correction removes that invalid assumption and saves the computed features
for an independent full comparison against the original prepared cache. It does
not change model inputs, rows, objective, batches or seeds.

`reaction_attention.resume` is restricted to this observed pre-optimizer failure.
It requires dead controller/worker identities, exactly the log and graph artifact
in the failed worker directory, no second-seed work, an unchanged plan and an
idle GPU. It archives that worker directory, preserves both previous leases, and
creates an exclusive recovery lease. No completed fit is repeated. This is an
implementation repair, not another transient infrastructure retry or scientific
negative result. Use `--execute` only after its read-only check passes.

## Implemented, not a result claim

Source `9a06d04` contains the registered four-arm comparison, separate reactant,
product and collider metadata, affinity features, train-only scaling, fixed pooling,
two-head cross-attention, existing GPU trainer integration, state-only inference,
live timing and protected/independent scoring adapter.

The pinned server environment passes27 focused tests; four historical GPU-training
tests were intentionally skipped with CUDA hidden during the final CPU verification.
All seven new module tests pass, including neural forward/gradient checks on CPU.
This is verification, not CPU model training. An earlier broad test invocation lacked
the required CUDA workspace variable and failed those four historical GPU tests;
it was not a chemistry experiment and no fit completed. No other job was stopped.

Additional real-mechanism verification used16 existing training states, no optimizer
steps. All four zero heads reproduce the base exactly. The mechanism remains59
species/356reactions/rank54; affinities are finite. Fixed-pooling heads have20,449
parameters and attention heads22,561. Thus attention includes2,112 extra parameters;
do not call this an equal-capacity comparison.

Separate standards/specification review found no blocking model defect. Before
launch we removed an unenforced whole-worker time ceiling, used the public argparse
choices interface, and added independent collider/pooling and common-initialization
checks. Finite optimizer work and per-fit1800s deadlines remain declared.

## Infrastructure result

VPN was restored with the existing connection helper. GPUs0 and1 were occupied;
their jobs were not stopped or altered. GPU2 was idle and selected for admission.
The standard protected isolation probe failed at `torch.cuda.is_available()` before
staging or training. Controller PID250669/startticks1056116239 is no longer alive.
The original lease and incomplete run directory remain as provenance. No fits or
candidate inference jobs are running. GPUs3,4,5 also failed the unchanged probe.

Minimal existing-code reproducer, after checking the selected device is idle:

```python
from pathlib import Path
from benchmarks.research_contract.sandbox import probe
p = Path("<project-root>")
r = p / "runs/offline-accuracy/reaction-attention-20261014"
probe(p, r / "probe-input", r / "probe-output", gpu=2)
```

The sandbox contains `/dev/nvidia2` plus control/UVM nodes, and passes
`CUDA_VISIBLE_DEVICES=2`. A diagnostic-only process with these identical device,
filesystem and network mounts reports zero CUDA devices. Changing only the visible
selector inside that diagnostic to `0`, or to the physical device UUID, gives one
device and a successful four-element CUDA sum. No candidate fit was run under this
modified selector. This isolates host-index versus isolated-device enumeration,
not missing hardware, network access, reference permissions or model code.

`benchmarks/research_contract/sandbox.py` and its lock are owner-controlled. Neither
was changed. Training is blocked pending authorization for a narrow GPU-binding
repair, ideally resolving the physical GPU to its UUID while preserving the same
one-device mount. That repair must rerun the complete original isolation probe,
add a nonzero-host-device regression check, and retain all label/network/read-only
checks. It must not weaken scoring or open confirmation.

## Resume

1. Do not reuse or delete the original lease. It records a failed preflight, not a
   running or completed fit.
2. Obtain owner authority for the protected infrastructure repair. Preserve the
   original lock as provenance and use the authorized supervisor update process.
3. Confirm no running process/lease, rerun the pinned tests and isolation checks.
4. Register the one allowed infrastructure retry with a new output and lease,
   without changing arms, seeds, work or scientific gates. The existing controller
   rejects an existing output; do not bypass that guard or steal the old lease.
5. Run all four arms/both seeds, full U/D replay and `reaction_attention.review`.

Continued GBCT remains the incumbent. The earlier strict-consensus candidate remains
pending authorized promotion. This implementation has no measured accuracy gain.
