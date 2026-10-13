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

`research/` owns the persistent signed-increment mission, prospective evaluation
contract, campaign authority, evidence ledger and restart index.
`benchmarks.research_contract.replay -> adapter/reference/gates -> NumPy` scores
saved arrays independently of candidate training, using historical checked data
and selection readers. Admission is separate from numerical scoring. Runtime
isolation and confirmation custody are required boundaries, not properties of a
hash file. Historical trainers and production modules remain unchanged.

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

`offline_accuracy.paper_baseline.plan -> run -> fit -> coordinates` owns the
reduced-data Fuel source-recipe replication. It reuses `refinement.run.inputs`
and the paired runtime, acceptance and independent-verification functions.
It changes normalization, output count and training schedule, not the frozen
dataset or pass rule. Its result summaries feed the research review; original
study scripts and environments remain read-only. `paper_baseline.campaign`
adds a bounded 50k preparation/audit stage; `paper_baseline.data` gates training
on a preserved original training prefix and byte-value-identical development
arrays. It depends on `flame_conditioning.prepare/audit_labels/data` and stops
before training on any reference or identity failure.
`paper_baseline.review -> fuel_* queries -> report-content/FuelBaseline.jsx`
adds verified evidence to the same report, using the shared method registry.
`paper_baseline.scale_review -> fuel_scale_* / fuel_label_throughput -> FuelScale.jsx`
adds verified size/work and CPU-label-throughput tables without replacing the
historical 10k queries. The report never launches work or reads sealed test data.
`paper_baseline.matched_work.review -> plan / paper_baseline.review.primary`
feeds `fuel_matched_*` queries into `FuelMatchedWork.jsx`. It requires the
eight-result campaign verification and preserves prior report queries.
`paper_baseline.matched_work.campaign -> run -> fit/plan` owns the equal-update
size experiment. It reuses the checked nested data, coordinate adapters and
independent verification, but freezes 50k preprocessing for both pools. Its
summaries remain separate from the original fixed-epoch evidence.
`matched_work.budget_campaign -> run / budget_checks` extends only the work
budget on the checked 200k pool; `budget_review -> fuel_budget_* -> FuelBudget.jsx`
retains the earlier evidence. It changes neither chemistry labels nor production code.

`paper_baseline.matched_targets.campaign -> run -> fit/model/coordinates/plan`
owns the 200k four-target, three-objective, two-seed comparison. It reads checked
data through `matched_work.run.inputs` and uses explicit adapters for
`paper_baseline.run.evaluate_and_verify`. The latter keeps its original defaults.
Coordinate code reuses stable flame/paired/improve transforms. Campaign checks
bind all results, common inputs/initial weights and objective-paired warmup hashes.
No production code depends on this experiment; review is a one-way artifact consumer.
`matched_targets.review -> matched_targets_* -> report-content/MatchedTargets.jsx`
binds the complete factorial results into the same review while preserving history.

`paper_baseline.adaptive.launch -> campaign -> run -> fit -> model/coordinates/plan` owns
the six-fit residual-calibrated local correction experiment. It reads the pinned
matched-target baseline and `matched_work.run.inputs` data gate. It reuses paired
runtime/metrics and explicit `paper_baseline.run.evaluate_and_verify` adapters.
Each checkpoint contains its frozen base and train-only scale tables; inference
never reads reference labels. Historical trainers and production code do not
depend on this module. The method catalogue names its three controlled arms.

`paper_baseline.decoder_loss.campaign -> run -> fit/model/coordinates/diagnostics`
isolates base-relative linear versus fixed-crossover asinh correction and log
versus Huber normalized error. It reads pinned adaptive baseline identities,
reuses adaptive head/features/preprocessing and matched data gates, and supplies
explicit prediction adapters to the independent evaluator. Its no-training
FP32 oracle audit and rare-event diagnostics are artifact consumers, never
inference inputs. Older adaptive runs and production modules remain unchanged.

`paper_baseline.resolved_fit.run -> data/fit/plan` owns the small training-only
fit diagnostic. It reuses checked training audit references, pinned GBCT bases,
adaptive features and fixed-asinh decoder arithmetic. Direct-coordinate MSE
and physical-log objectives share resolved component masks and two head sizes.
Development and test labels do not enter fitting or scores. Its saved artifacts
feed review only; it does not replace the base or any full-data campaign.

`paper_baseline.correction_scaling.prepare -> coverage/model/plan` freezes continued
GBCT bases, checks label provenance, and caches training-only targets/features.
`run -> train/evaluate/model` owns the separate small regression gate and matched
nominal-label size progression. It reuses adaptive features, decoder_loss arithmetic,
resolved_fit's probe and matched_work data gates without altering historical code.
Coverage and group reports feed review; no production dependency points back here.
`correction_scaling.review -> saved predictions / prepared caches` independently
recounts acceptance, checks paired recipes and records initial squared-loss
concentration. Its progression decision does not launch another experiment.
`correction_scaling.tail -> checked caches / frozen base / saved MSE head` adds
read-only correction-tail and constant-shift diagnostics. `train -> objectives`
adds a separately named coordinate Smooth L1 arm without changing old objectives.
`robust_review -> review.recount` verifies paired controls and preservation.
`run -> gate` requires a hash-bound, recomputed passing preceding rung above 10k;
no review module launches training. New evidence stays separate from old runs.

`paper_baseline.correction_transfer.prepare -> selection / correction_scaling`
freezes same-source unseen rows and targeted tail/control audit states.
`evaluate -> metrics / frozen correction_scaling models` measures transfer,
preservation and fixed damping without fitting. `audit -> flame chemistry`
rechecks selected fixed-T/V references using bounded CPU integration. These
artifacts never overwrite training labels or promote a correction policy.

`paper_baseline.hidden_features.prepare -> checked correction caches / frozen base`
stages train-only labels and state-only evaluation inputs. `worker ->
correction_scaling.train/objectives + hidden_features.model` tests frozen feature
access against a near-parameter-matched head. `campaign -> research_contract
admission/sandbox/scorer` isolates candidate execution and scores returned arrays.
The trusted controller owns references; candidate inference cannot mount them.
Confirmation remains separately disabled. Production modules remain unchanged.
The same isolated worker runs a separately registered preservation comparison;
`hidden_features.preservation -> correction_scaling.objectives` adds only a
training acceptance-interval penalty through an explicit trainer callback.
It never changes scoring or the reference-free inference path.
`hidden_features.resume -> campaign worker/admission + checked artifacts` reconciles
stopped controllers by PID/start time and reuses complete fits. It preserves the
interruption record and never overwrites partial evidence or kills foreign work.

`paper_baseline.local_predictability.run -> research_contract identity/sandbox +
hidden_features resource monitor` owns a read-only local-variation diagnostic.
`neighbors -> PyTorch` constructs graphs from features and IDs, without labels;
`analysis -> pinned reference scorer + stable decoder` measures full-target and
residual transfer on those same graphs. It cannot promote a model or change data.
`local_predictability.review -> saved arrays` recounts physical outcomes;
`summary -> reviewed report` exports aggregate evidence without state arrays.

`paper_baseline.species_distance.run -> verified local_predictability artifacts`
reuses the fixed libraries and queries. `metric -> training-only rank statistics`
fits species-specific diagonal weights without query labels. `worker ->
local_predictability.neighbors` searches isolated feature-only inputs; `analysis`
uses the same graph for full-target and correction transfers. The existing
independent review recounts physical scores. No evaluator or production dependency
changes, and this diagnostic cannot promote a predictor.

`paper_baseline.local_function.run -> verified species-distance artifacts +
research_contract sandbox/gates` stages matched fixed-centre models. `worker ->
model + correction_scaling base/adaptive features` fits coefficients on training
labels and predicts from state-only mounts. `review -> saved predictions` performs
an independent recount. No production module or protected evaluator is changed.

`offline_accuracy.paired.plan -> run -> fit -> coordinates` owns the GBCT target
adaptation and paired increment/state error-scale experiment. It reads the frozen
dataset through `refinement.run.inputs`. `paired.verify` replays artifacts and
independently checks both acceptance policies and physical diagnostics. Small
verified summaries feed review; historical and production modules do not depend
on this new experiment. See its local README for the predeclared matrix.
`paired.runtime` validates the project-local GPU environment and captures versions.
`paired.review -> paired_* queries -> report-content/PairedAccuracy.jsx` is the
one-way publication path. The canonical registry supplies its method names.

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
`parallel_labels.run -> storage/worker -> chemistry.EndpointIntegrator` adds
bounded process-parallel labels and immutable resumable chunks. It preserves
`sample_split` row order and the dataset reader interface. `parallel_labels.benchmark`
checks exact serial-label parity and throughput before expansion. Existing
datasets and GPU training remain unchanged; new data require a fresh audit.
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

The research-only local-function module uses `local_function.run` for admission,
`model` for frozen distance kernels, `worker` for Adam fits and state-only replay,
and `convex` for separately registered fixed-basis optimization controls. Both
paths share the existing isolation and protected scorer. `postcheck` and
`audit_metrics` independently verify saved predictions; they do not modify the
contract or production models. See the module README for interfaces.

`interval_calibration.prepare/model/worker` add a 58-constant, training-derived
calibration diagnostic. They reuse `local_function.run/review/postcheck` and the
same protected evaluator. Dependency flow is controller -> candidate module ->
existing base/decoder; production has no dependency on either research module.

`conditional_calibration` reuses the registered `gradient_probe.run` controller
and the interval-consensus learner for training-only cross-fit checks. It changes
scalar output resolution, not scoring. Its independent review does not open U/D
or confirmation. See its README for fold exposure and artifact interfaces.

`selective_correction.prepare/model/worker/review` tests a learned selector
between frozen predictions. It uses the same diagnostic controller and the
historical checked correction loader. The selector adds no references to its
inference inputs. Full candidate qualification remains a separate controller path.

`species_heads` reuses the same row-staging and trainer with an explicit model
factory. Its batched heads have disjoint species parameters, not routing. Its
state-only inference mount excludes the mechanism and labels. Both new and old
trainer paths use the existing default model factory unless explicitly supplied.

`kinetic_features` reuses the diagnostic controller, correction trainer and
initial-rate formulas. Its separate state-only inference worker receives the
private mechanism and frozen weights/scaler, not reference arrays. The trusted
review process scores full saved outputs. No production dependency is added.

`paired_base` reads pinned prepared caches through the research-contract adapter,
aligns a common training-only population, and uses the protected scorer plus an
independent recount. It has no training or production dependent. Cached mean
predictions do not establish fresh inference isolation or runtime cost.
Its separately registered full screening adapter uses `gradient_probe.run`
admission -> `hidden_features.run_worker` -> state-only `screen_worker` with two
pinned models. `screen_review` independently verifies arrays and runs protected
gates. There is no production or automatic-promotion dependency.

`element_projection` uses the same guarded diagnostic controller and state-only
base replay. A mechanism-derived constraint matrix is its only added metadata.
It performs no training. Its declared projection/fallback is scored as returned,
with raw base predictions retained; no protected evaluator changes are made.

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
