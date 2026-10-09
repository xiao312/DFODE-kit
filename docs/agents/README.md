# Shared Operational Docs

This directory contains deeper operational docs used by both humans and coding agents.

Use this docs tree for information that is too detailed or too volatile for `AGENTS.md`.

## Documents
- `init-cli-spec.md`: pointer to the shared canonical init CLI reference in `docs/init.md`
- `verification.md`: required local/CI verification loop
- `worktrees.md`: parallel branch + worktree workflow
- `roadmap.md`: near-term harness and refactor priorities
- `topology.md`: documentation and repo topology guidance
- `ci-tests-plan.md`: narrow next-step plan for harness parity and docs invariants
- `cli-usability-plan.md`: CLI agent-usability improvement plan
- `data-io-contract-plan.md`: data I/O contract and refactor plan
- `train-config-plan.md`: training/config refactor plan for experiment throughput
- `package-topology-spec.md`: target package organization and module-boundary spec
- `package-topology-migration-plan.md`: staged migration plan toward the target package topology
- `cfd_conditioned_data.md`: read when working on CFD-conditioned trajectory datasets
- `precision-conditioning-research.md`: original precision-audit stages and review workflow
- `cantera-tolerance-sources.md`: read before changing the numerical tolerance ladder
- `fuel-mechanism-source.md`: initial mechanism identification and provenance gate
- `okafor-mechanism-provenance.md`: read before claiming original-release equivalence or publishing the recovered mechanism
- `fuel-cfd-benchmark-alignment.md`: approved shift from reactor diagnostics to flame-based evaluation
- `flame-source-and-runtime-contract.md`: read before reusing study assets or preparing a copied CFD case
- `representation-accuracy-success-sources.md`: source checks for offline tolerance criteria, pressure sampling, and SSPI

## Philosophy
- `AGENTS.md` is the entrypoint for repository workflow.
- Documentation should be shared across humans and agents whenever possible.
- New durable rules should prefer tests, CI, or scripts over prose alone.
