# Fuel baseline replication: source recipe and limits

Checked 2026-10-09. This note separates manuscript facts, saved study-code facts,
and proposed adaptations. Reading source code is not a reproduced result.
Original study files and remote environments were read-only during this audit.

## Decision

Build stronger **Fuel source-recipe baselines** before adding more target families.
Use the two recovered training recipes, not another short cosine schedule.
Start on the fixed offline development domain so that training changes can be
checked without changing the reference labels. Then increase data size as a
separate factor. A small-data source-recipe run is not a full paper reproduction.

Keep the [method catalogue](../research-method-catalogue.md) as the source of
method names. Keep the [dual-scaling protocol](representation-accuracy-success-sources.md#required-dual-scaling-protocol-for-later-runs)
as the source of scoring rules. This note defines provenance, not a new pass rule.

## 1. What the Fuel manuscript specifies

The study uses a 60/40 NH3/CH4 premixed flame, equivalence ratio one, 300 K,
and 1 atm. Its 500 cells and 2,500 sampled microsecond steps yield about
1.25 million states. Interpolation and constrained perturbations produce an
eight-million-state dataset. Perturbations use ±100 K, pressure-span scaling,
and species exponents between 0.85 and 1.15. Temperature, nitrogen and heat
release filters restrict accepted states. The network has four 800-unit layers,
standardized inputs/outputs, and no argon output. Two targets are compared:
transformed-state differences and signed-power physical increments, both with
exponent 0.1. [Fuel manuscript, Sections 2.2–2.4](https://arxiv.org/html/2507.08277v2)

The evaluation separates tiny increments at 1e-15 from other increments. SSPI
checks whether tiny references also have tiny predictions; a-indexes measure
relative-error fractions for the other group. The CFD policy uses the
transformed-state model above 1000 K, direct power between 305 and 1000 K,
and zero below 305 K. Thus, paper a10 and our all-component mixed-budget
acceptance are not the same metric. [Fuel manuscript, Sections 3.2–3.3](https://arxiv.org/html/2507.08277v2#S3.SS2)

## 2. What the recovered training code specifies

These are user-owned source files on the study host. They were read directly
again for this note, and their hashes still match the earlier
[source inventory](flame-source-and-runtime-contract.md#original-training-controls-text-only-inspection).
Use portable root names from that inventory; keep absolute machine paths in
ignored run manifests.

**Source A — transformed-state run**

`${WORKSPACE}/active_work/training/train_800_800_800_800_loss1_dataset_1Dflame_1d_flame_60nh3.ULFS_0.0-2.5ms_interpolate_perturbated_heat_release_filtered/singleMLPTrainingCV.asinh_loss.py`

SHA-256: `b85a0e704e314a95e792583f01be8c79d8229c4e20cadff7f49a76b32a04d0d8`.

**Source B — direct-power run**

`${DFODE_STUDY_ROOT}/test_runs/test_250903_151713/train_mlp_dev_target_transform_power.py`

SHA-256: `05ec1ff438531b4338c3fedf315905cf924dd7d81bd37a62b808769ff64ab7f6`.

`DFODE_STUDY_ROOT` means the separate existing `dfode_project` study workspace.
Its location is not the current DFODE-kit worktree. These source files are
primary evidence for the saved runs. We have not established that these exact
scripts and checkpoints produced every published figure.

### Network and chemistry

Both scripts use `61 -> 800 -> 800 -> 800 -> 800 -> 58`, GELU, a linear output,
FP32 tensors, and no active batch normalization. Inputs are temperature,
pressure, and 59 mass fractions in mechanism order. The outputs omit argon.
The source architecture has 2,018,458 trainable parameters. Both use ordinary
PyTorch Linear initialization and Adam defaults apart from the learning rate.
Both optimize mean absolute error in normalized coordinates. Other diagnostic
losses in Source A do not enter its backward pass. Its filename does not mean
that it trains with asinh loss. **Evidence: Source A/B above.**

The recovered labeling scripts use a Cantera reactor with energy disabled,
fixed volume, a 1e-6 s interval, and tolerances `rtol=1e-6`, `atol=1e-10`.
The recovered mechanism has 59 species and 356 reactions. Its pinned hash is
`26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0`.
The script locations and chemistry interface are recorded in the
[original labeling and runtime audit](flame-source-and-runtime-contract.md).
These reactor constraints come from code inspection, not an assumption that
all combustion datasets use the same energy equation.

### Exact normalization

Let `B(y)=(y^0.1-1)/0.1` and `d=Y_after-Y_before`.

| Setting | Source A: transformed-state | Source B: direct power |
| --- | --- | --- |
| Input species | `B(Y)` | `B(Y)` |
| Temperature and pressure | Linear, then standardized | Linear, then standardized |
| Input center | Feature mean | Feature mean |
| Input scale | Sample standard deviation, `ddof=1` | Population standard deviation, `ddof=0` |
| Raw target | `B(Y_after)-B(Y_before)` | `sign(d)*abs(d)^0.1` |
| Target center | Component mean | **Zero**, not the component mean |
| Target scale | Sample standard deviation, `ddof=1` | Population standard deviation, `ddof=0` |
| Constant-feature guard | Present only as commented code | Zero standard deviation replaced by one |

**Evidence: Source A/B preprocessing.** Source B omits the factor `1/0.1` in
the paper's direct-power formula. With its standard-deviation scaling, that
positive constant cancels algebraically. This does **not** make its zero-center
convention identical to centered Z-score normalization.

Source A takes absolute values of stored endpoints. Source B clips stored
species endpoints into `[0,1]`. Both fit normalization before their validation
split. Do not import those repairs or the split leakage silently. Use validated
signed reference increments and train-only statistics in the new implementation.
Record these deliberate deviations.

### Exact training schedules

| Setting | Source A | Source B |
| --- | --- | --- |
| Epochs | 1500 | 2000 |
| Nominal batch size | 20,000 | 20,000 |
| Initial learning rate | 1e-3 | 1e-3 |
| Epoch order | Fresh NumPy permutation | Shuffled PyTorch DataLoader |
| Partial final batch | Dropped | Kept |
| Rate changes | Tenfold reductions after zero-index epochs 500 and 1000 | StepLR, step size 500, gamma 0.1, called after each epoch |
| Adam state at reduction | **Reset** | **Retained** |
| Script seed | 555 | 555, but set after the random split |
| Validation | Last 400,000 row positions | Random 95/5 row split |

**Evidence: Source A/B loops.** In Source A the change occurs after training
the specified zero-index epoch. With one-indexed epoch labels, rates are
1e-3 for epochs 1–501, 1e-4 for 502–1001, and 1e-5 for 1002–1500.
Source B uses 1e-3 for 1–500, 1e-4 for 501–1000, 1e-5 for 1001–1500, and
1e-6 for 1501–2000. Its final scheduler step has no further training effect.

Source B calls `random_split` before `set_seed(555)`. The supplied seed therefore
does not reproduce that old split by itself. Keep a new explicit split manifest.

The old Source A log records eight million rows, of which 7.6 million train.
At 20,000 rows per batch, this gives 380 updates per epoch, 570,000 updates,
and **11.4 billion training-row presentations**. This arithmetic follows the
source loop and the [recorded log inventory](flame-source-and-runtime-contract.md#original-training-controls-text-only-inspection).
The corresponding Source B recipe would give 760,000 updates and 15.2 billion
presentations on the same row count. Do not equate epochs with optimizer steps.

## 3. Data replication gaps

The recovered source mechanism and labeling operation are specific enough to
reuse safely. The complete published dataset lineage is not yet established.

- The raw paired sample files retain 2,401 times, not all 2,500 expected times.
- The existing raw NPY has 1.25 million FP64 rows, but its row positions do not
  preserve source-time IDs. Source text fields have about six significant digits.
- The inspected `sample_n_interpolation_aug_1d.py` has a default 1 K grid and
  computes interpolated states, but its active return is **the original sorted
  array**, not the combined array. It is evidence for a raw extraction path,
  not proof that the published interpolation used 1 K. Its hash is
  `099a066330d571a9451ff13c110b5dc12b183cb086b635c24a56535a685f0e44`.
- Exact published interpolation spacing, perturbation draw dependencies,
  normalization/closure order, numerical heat-release cutoff, and row-selection
  seed are not fully bound to the retained eight-million-row artifact.
- A saved training run is useful primary evidence, but file naming is not proof
  of published-checkpoint identity.

These observations use read-only source inspection and the
[asset inventory](flame-source-and-runtime-contract.md#flame-state-sources).
Resolve them before using the label **full paper reproduction**. Do not run the
legacy scripts unchanged: they write to old study paths and use large worker
counts. Reimplement their verified numerical recipe in the project run tree.

## 4. GBCT comparison

The paper specifies widths 1600/800/400, GELU, L1, Adam, and Z-score
normalization. Stage one uses 2500 epochs, batch 1024, and learning rate 1e-4.
Stage two uses another 2500 epochs, batch 262144, and learning rate 1e-5.
It applies signed power with exponent 0.5 to the rate of the Box–Cox state
difference, whose exponent is 0.1. These are different targets and training
conditions from the Fuel recipe. [GBCT manuscript, Section 3](https://arxiv.org/html/2512.05685v1#S3.SS2)

The pinned released code has discrepancies that prevent treating its default
command as the paper protocol:

- `batch_grow_rate` defaults to 128, not the paper's factor 256.
- The chemical loader scales centered labels by their uncentered mean absolute
  value, not their standard deviation. It returns FP32 tensors and normalizes
  before splitting. See the [existing loader audit](gbct-source-and-adaptation.md).
- The single-device trainer assigns `inputs_train` and `labels_train`, but its
  batch loop reads undefined `inputs_data` and `labels_data`.
- The distributed loop computes batch count from the original batch size while
  its slices use the increased size. Its explicit epoch shuffle is commented.
- Rate/batch changes reset Adam at the start of epochs divisible by 2500,
  including epoch 5000. This is not precisely the prose's two equal phases.

These are static observations, not executed failures. Sources pinned at
`982e58954af5c5e8e3860d3057363b83bcbaeac6`:
[configuration](https://github.com/Seauagain/GBCT/blob/982e58954af5c5e8e3860d3057363b83bcbaeac6/deepode/nn/config.py),
[trainer](https://github.com/Seauagain/GBCT/blob/982e58954af5c5e8e3860d3057363b83bcbaeac6/deepode/nn/trainer.py).

For a bounded comparator, select the **paper-intent** schedule explicitly and
use train-only Z-score statistics. Keep the same NH3/CH4 labels and split.
Call this a GBCT paper-recipe adaptation, not reproduction of the original
chemical benchmark. Keep an explicit optimizer-reset policy. Do not spend this
campaign repairing or running the upstream trainer.

## 5. Difference from the completed paired campaign

The [paired campaign](../../benchmarks/offline_accuracy/paired/README.md) used
10,000 training states and 1,023 development states; batch 256; 4,000 updates;
two cosine phases; and train-only centered population scaling. It used 59
network outputs while excluding argon from loss and scoring. The fixed physical
pressure center/scale was 101325/5066.25 Pa.

The training-row exposure was `4000*256 = 1,024,000`, about **11,133 times
less** than the historical Source A run. The number of unique training rows
was 760 times smaller. These ratios describe different factors; neither proves
that more work alone will recover the published performance.

The current [offline dataset](../../benchmarks/offline_accuracy/dataset.json)
uses four training snapshots, two development snapshots, independently sampled
0.95–1.05 atm pressure, and Cantera 3.2.0 references at `rtol=1e-12`,
`atol=1e-21`. Only a 16-state subset has the independent increment check.
Its wider pressure domain, explicit split lineage, tighter references, and
stable FP64 reconstruction are deliberate adaptations. Preserve them during
the first training-recipe comparison so the labels do not change at the same time.

## 6. Bounded implementation plan

1. Implement the two Fuel recipes as separately named configurations. Match
   the 58-output architecture, each normalization convention, L1 loss, epoch
   traversal, and each Adam/schedule rule. Use the current GPU environment and
   record versions, driver, precision flags, seed and source hashes.
2. Keep the same 10k row IDs per seed and the same development rows. Use two
   retained seeds, not the better seed. With fewer than 20k rows, explicitly
   cap the effective batch size at the training count. This is full-batch
   optimization, not a literal reproduction of the original minibatch noise.
   Source A would otherwise perform zero batches. State this adaptation.
3. Save fixed epoch checkpoints and training/development curves. Report epochs,
   optimizer updates, effective batches, row presentations, wall time and peak
   memory together. Never describe 1500 full-batch updates on 10k rows as
   matching 570k updates on 7.6 million rows.
4. Keep validation data out of normalization and training. Seed the new split
   explicitly. Preserve constant-feature guards, stable inverse calculations,
   and visible invalid-domain counts. These are intentional safety differences.
5. Report the unchanged increment and state acceptance grids, whole-state
   acceptance, error tails and physical checks. Add paper-style SSPI and
   relative-error fractions with explicit masks as supplementary metrics. A
   good tiny-increment score must not replace general accuracy.
6. If this establishes a useful training improvement, grow to 50k and 200k
   accepted states with preserved source groups and a measured cost limit.
   On these sizes the 20k batch again gives multiple updates per epoch. Keep
   the data-size effect separate from the optimization-budget effect.
7. Assess the paper's fixed temperature-switch policy as a named offline
   composition after the two models exist. Specify exact equality handling
   at 305 and 1000 K before scoring. Do not tune the switch on development
   results and call it the paper's policy. CFD remains a later stage.

The first run can answer whether the source training recipe gives stronger
baselines on our fixed domain. It cannot establish eight-million-state
reproduction, independent-case generalization, solver-level error control, or
CFD speedup. Those claims require separate evidence.
