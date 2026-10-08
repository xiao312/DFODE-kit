# Architecture

## Current repository structure

- `dfode_kit/cli/`: CLI entrypoints and subcommands
- `dfode_kit/cases/`: explicit case init, presets, sampling, and DeepFlame-facing helpers
- `dfode_kit/data/`: contracts, HDF5 I/O, integration, augmentation, and labeling utilities
- `dfode_kit/models/`: model architectures and registries
- `dfode_kit/training/`: training configuration, training loops, registries, and preprocessing
- `docs/agents/`: agent-facing operational and planning docs
- `tests/`: lightweight repository and harness tests

## Precision conditioning research branch

`benchmarks/precision_conditioning/` owns the standalone representational audit.
It depends only on NumPy and does not alter production data or training paths.
`tests/test_precision_conditioning_audit.py` verifies its numerical invariants.
Optional rendering flows from audit JSON through `plot.py` (Matplotlib) and
`build_report.py` (standard library) into `publish_pages.py` (standard library).
The publisher produces static scientific artifacts only; deployment is a separate
Git operation on `research-pages`. `tests/test_precision_conditioning_pages.py`
checks its dry-run boundary and generated files. Wiki, Discussions and Projects
link the evidence without becoming numerical inputs.
The staged research and review plan is in
`docs/agents/precision-conditioning-research.md`.

Checkpoint 02 is isolated in `benchmarks/precision_conditioning/reference/`.
`pilot.py` calls `chemistry.py` (Cantera and SciPy), writes raw records, then calls
`analysis.py` (NumPy). `targets.py` uses NumPy only and does not depend on the runner.
Selected results flow to the existing GitHub review surfaces. Production labeling
and training do not depend on this pilot. Its local README defines the run contract.

## Current refactor themes


### 1. Harness engineering
The repository now includes:

- `AGENTS.md`
- local verification commands
- lightweight CI
- documentation topology for agents and maintainers

### 2. Data contracts and workflow boundaries
A contracts layer is used to make HDF5 dataset assumptions explicit and testable.
The canonical `dfode_kit.data` package now also owns the main data-preparation boundary:

- HDF5 sampling outputs
- HDF5-to-NumPy conversion
- perturbation-based augmentation
- CVODE/Cantera labeling
- integration utilities used by downstream workflows

### 3. Config-driven training
The training stack is moving toward explicit config objects and registries so new model architectures and trainer types can be added without editing a monolithic training loop.

### 4. Agent-friendly CLI
The CLI now uses lighter command discovery and deferred heavy imports for improved usability in minimal environments.

## Architectural end state of the recent refactor

The repository has now completed the transition away from the older compatibility layout. In particular, these legacy layers are removed from `main`:

- `dfode_kit/cli_tools/`
- `dfode_kit/df_interface/`
- `dfode_kit/data_operations/`
- `dfode_kit/runtime_config.py`
- legacy `dfode_core` model/train compatibility packages

The current published docs should therefore treat `cli`, `cases`, `data`, `models`, `runtime`, and `training` as the only canonical implementation homes.
