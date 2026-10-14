# Protected numerical research contract

## Responsibility and non-goals

Score aligned signed-increment arrays and apply explicit, conjunctive development
gates. Reproduce the historical bases and rejected corrections. This module does
not train, select a checkpoint, relabel data, promote production, or open a test.

## Interface and inputs

`reference.evaluate_predictions` is the imported NumPy-only reference scorer.
The adapter enforces the fixed tolerances, row/species identity and input hashes.
`gates` compares each candidate to both the fixed anchor and moving incumbent.
Missing required evidence is inconclusive, never an implicit pass. Nonfinite
predictions remain in denominators and prevent promotion.

`python -m benchmarks.research_contract.replay --project <root>
--bindings research/bindings.json --output <new-directory>` checks identities.
Add `--execute` to write an independent real-array score report. It reads the
saved predictions; neural label-free replay remains a separate required check.
The output is a new JSON artifact; existing reports and models stay unchanged.

`sandbox.command` supplies a Bubblewrap allowlist for candidate workers: source,
the pinned Python environment, one input directory and optional model directory
are read-only. Only that worker's output directory is writable. Home, credentials,
reference directories and network interfaces are absent. `sandbox.probe` tests
these boundaries and a GPU kernel before execution. The controller remains trusted;
this is candidate-process isolation, not separate-custodian confirmation security.

Owner-authorized repair (2026-10-14): select the mounted physical GPU by its
host-resolved UUID, not by its host ordinal after isolated enumeration. The device
mount allowlist and every read-only/network/label boundary are unchanged. Invalid
or ambiguous UUIDs fail closed; the probe now also requires exactly one visible
CUDA device. The old lock remains in Git and failed-run artifacts. Only the sandbox
entry in the current evaluator lock changes; scoring and admission rules do not.

## Dependencies and security

The standalone scorer depends on NumPy. The replay adapter uses the historical
checked artifact readers and frozen row selections. Candidate inference must be
isolated from evaluator labels. This module alone does not supply an OS boundary.
Runtime admission must verify that boundary before autonomous candidate execution.
No credentials belong in this module or its artifacts.

Dependencies: `replay -> adapter/reference/gates -> NumPy`; historical readers are
read-only dependencies, never consumers of this new contract. The authoritative
scientific and operating specification is `research/`.

## Verification

`python -m pytest tests/test_research_contract*.py -q` checks synthetic scoring,
empty groups, row-order rejection, two-comparator guardrails and budget admission.
The original 22 synthetic tests are retained. Server real-array checks must
reproduce base/MSE/Smooth L1 counts and reject their known regressions.
