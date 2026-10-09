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

The bounded learning comparison lives in `benchmarks/precision_conditioning/learning/`.
`prepare.py -> reference.pilot/chemistry/analysis`; `train.py -> dataset.py and
reference.targets -> PyTorch`; `review.py -> saved evidence -> static Pages and
issue attachments`. It owns checked group splits and fixed model budgets. It does
not change production preprocessing, model registries or training behavior.
`learning/fit_probe.py -> saved training arrays/preprocessing and train.network`
is a read-only scientific fit check. It does not score the held-out test set.
`learning/fit_diagnostic.py -> train.preprocessing/network and metrics.evaluate`
reuses only training rows for a bounded loss/update/subset diagnosis.
`fit_review.py -> saved fit evidence -> static Pages`; `verify_fit.py` independently
replays the saved diagnostic models. No production module depends on these tools.
`learning/polish.py -> polish_core.py -> train and fit_diagnostic` adds bounded
multi-seed optimization with an all-component stop/save gate. `verify_polish.py`
replays selected models; `polish_review.py -> saved results -> static Pages` is
the one-way report dependency. No held-out data enter this training-fit stage.

## Flame-conditioned research

`offline_accuracy.refinement.plan -> run -> fit -> tolerance coordinates`
adds isolated longer-training, physical-loss and frozen-residual experiments.
It reads checked flame datasets and original models without changing them.
`refinement.verify -> saved models + independent acceptance counts` supplies
small evidence to the existing report. No production or historical trainer
depends on this module; its README freezes the comparison and cost boundaries.
`refinement.review -> verified summary JSON -> refinement_* snapshot queries ->
report-content/Refinement.jsx` preserves all earlier report evidence and IDs.

The later offline-accuracy stage is owned by `benchmarks/offline_accuracy/`:
`declared pressure domain -> flame prepare/audit -> matched final-checkpoint
training -> physical tolerance/SSPI metrics -> acceptance/cost review`.
It reuses the existing numerical and model modules. CFD transfer is a later,
separate evaluation, not a dependency of this representation comparison.
`offline_accuracy.review -> verified evaluation summaries -> existing report
snapshot -> report-content/OfflineAccuracy.jsx` is a read-only evidence path.
It preserves prior queries and has no dependency back into training.

Post-score pressure diagnosis is a separate evidence path:
`pressure_diagnostic -> frozen models + chemistry -> new paired reference/prediction
artifacts -> verify_pressure_diagnostic (independent physical recomputation) ->
review_snapshot -> report`. It cannot write to the frozen test or model directories.
There is no dependency from production solvers or training on this diagnostic.

`benchmarks/flame_conditioning/` owns the application-aligned comparison separately
from the homogeneous-reactor audit. Its one-way dependency is `read-only original
cases -> extracted state/lineage files -> fixed-temperature, fixed-volume chemistry
labels -> target comparison -> offline/coupled evaluation -> static review`.
`chemistry.py` depends on pinned Cantera/NumPy/SciPy. It does not alter production
labeling, solver installations, or the earlier reference formulation. The local
README defines the provenance, split, compute and secret boundaries. No production
module depends on these experimental tools.
`extract.py -> paired line samples` establishes explicit source identities;
`prepare.py -> augmentation.py/chemistry.py` preserves snapshot splits and writes
bounded label arrays. `scout.py -> chemistry.py` independently checks a selected
subset. The fixed-T/V numerical module does not reuse the adiabatic RHS.
`coordinates.py -> reference.targets` reuses signed-power/asinh primitives and
adds stable transformed-state differences. `data.py -> checked dataset/chemistry`
validates hashes and lineage. `metrics.py -> saved physical predictions` evaluates
non-argon accuracy and all-species conservation without modifying predictions.
`audit_labels.py -> data/chemistry/scout` checks augmented labels;
`train.py -> data/coordinates/metrics` requires that audit and exposes validation
only. `verify.py -> saved models/coordinates/data` independently replays the
matched-budget result. Held-out 2D evaluation and CFD deployment are later layers.
`heldout.py -> frozen training artifacts/reserved snapshot/chemistry` prepares a
separate labeled test. `scout.py -> saved test states` audits its references.
`evaluate_heldout.py -> audited test/verify/metrics` checks frozen model hashes and
scores uniform and temperature-balanced populations separately. Test evidence has
no dependency back into training or checkpoint selection.
`verify_heldout.py -> frozen plan/saved test predictions/verify_physical`
independently reconciles both test populations and the complete model list.
`verify_scaling.py -> completed dataset artifacts/data` checks the nested training
and identical validation contract before comparing dataset sizes.
`copy_case.py -> inspected original fields/mesh` prepares an allowlisted isolated
restart and depends on the canonical case's inactive spray dictionary. Execution
uses the existing image, outside the preparation command. It
neither calls the original case scripts nor modifies source cases or installations.
`review_cfd.py -> copied fields/logs/mesh/original geometry/preparation manifest` checks completion and
rehashes the allowlisted original files through a read-only mount.
`compare_cfd.py -> review_cfd/copied case manifests/fields` compares two completed
CVODE-only restarts with identical inputs and distinct explicit tolerance presets.
`runtime_parity.py -> validation artifacts/installed Cantera 2.6` checks a bounded
fixed-T/V subset without importing the research training environment.
`review_snapshot.py -> small saved model/audit/CFD JSON` emits sanitized reviewed
data for the downstream HTML report. Presentation cannot alter source evidence.
`flame_conditioning/report-content -> sanitized snapshot/shared Data app API`
owns the review narrative and charts; the shared runtime compiles them for Pages.
Its pure `policy-bins.mjs` helper presents fixed BC/PT/hybrid/zero policy columns
from reviewed bin rows and rejects duplicate or unequal-population comparisons.
The report has no dependency back into model training or reference generation.
`checkpoint_conversion -> allowlisted historical checkpoint/patched isolated Torch`
`historical -> checked numerical NPZ/Torch inference/coordinate reconstruction`
`historical_validation -> historical/data/metrics/fixed hybrid policy`
`heldout -> frozen training and historical artifact identities`
`evaluate_heldout -> historical and new predictors/common physical metrics`
`verify_physical -> saved predictions/data/independent Cantera-density calculation`
`verify -> saved-model replay/optional independent training physical metrics`
`filter_audit -> audited labels/formation enthalpies/saved validation predictions`
exports checked numerical arrays. Historical adapters must depend on this NPZ
boundary, never on legacy pickle loading in the working solver environment.

## Current refactor themes

The review-only method registry has one-way dependencies:
`method-catalogue.mjs -> report selection labels / ReviewGuide / Markdown renderer`.
The renderer generates `docs/research-method-catalogue.md`, which GitHub review
surfaces link to. Executable experiment plans remain the authority for numerical
configuration. The registry never changes saved run IDs, predictions or scores.
The split guide reads existing verified report queries; it does not train or test.

The isolated `offline_accuracy.improve` benchmark adapts acceptance-directed
losses and local approximation methods. Dependency direction is
`plan -> run -> neural/local -> coordinates`, with read-only dependencies on
`refinement.inputs/loaders`, offline metrics and independent verification.
The later `arrhenius` candidate owns partial-pressure input features, independent
species heads and a local-table alternative. `diagnostics` reads checked prediction
artifacts; `review` binds their small state-error summaries into the existing app.
`improve.recheck -> saved predictions/run.verify` repairs only the documented
heat-diagnostic cancellation failure, with original results retained. Shared
physical metrics now contract increment errors before heat-release aggregation;
the independent Cantera checker remains separate.
`improve.physics_prior -> Cantera kinetics/dataset/audit/metrics` supplies explicit
rate and frozen-exponential non-learned controls. No production solver depends on
these fixed-step diagnostics; mechanism evaluation is included in their cost.
Verified small summaries feed GitHub review; production training and solvers
do not depend on this research path. See its module README for the fixed matrix.


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

The research Pages publisher can consume the reference pilot's static report,
figures, summary and analysis provenance. This is a one-way artifact dependency:
`reference analysis/renderers -> publish_pages -> research-pages deployment`.
It does not import the chemistry runner or change production training behavior.

The repository has now completed the transition away from the older compatibility layout. In particular, these legacy layers are removed from `main`:

- `dfode_kit/cli_tools/`
- `dfode_kit/df_interface/`
- `dfode_kit/data_operations/`
- `dfode_kit/runtime_config.py`
- legacy `dfode_core` model/train compatibility packages

The current published docs should therefore treat `cli`, `cases`, `data`, `models`, `runtime`, and `training` as the only canonical implementation homes.
