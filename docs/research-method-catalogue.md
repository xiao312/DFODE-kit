# Chemistry method catalogue

> Generated from [the canonical registry](../benchmarks/flame_conditioning/report-content/method-catalogue.mjs). Do not edit this page by hand. Run `node scripts/build_research_method_catalogue.mjs --write` after changing the registry or renderer.

This is the naming and diagram reference for the offline representation study. The [live report](https://xiao312.github.io/DFODE-kit/flame-conditioning/) contains measured results and the split/tolerance review guide. [Issue #3](https://github.com/xiao312/DFODE-kit/issues/3) records decisions. Old comments and artifact IDs remain historical records; use the mappings here to read them.

The [accuracy protocol](agents/representation-accuracy-success-sources.md#required-dual-scaling-protocol-for-later-runs) distinguishes CVODE tolerance parameters, state-based error scaling, increment-based error scaling and custom magnitude-dependent budgets. It records the earlier dual-scaling study. The current signed-increment campaign uses its protected [evaluation contract](../research/EVALUATION.md); see [current results](../research/RESULTS.md) for its unchanged rule and decisions.

## Three different concepts

1. **Representation:** the coordinate system for a signed physical increment.
2. **Method recipe:** input features, representation, approximator, loss and optimization.
3. **Run:** recipe plus dataset, split, seed, precision and work budget.

Changing a seed does not create a new method. Sharing an asinh transform does not make two recipes identical. The historical `arrhenius-*` IDs mean inverse-temperature and log-partial-pressure features, not an exact Arrhenius law. A matched GBCT target adaptation is implemented; full GBCTNet reproduction, ISAT and spectrum-informed MSNN are not. See the [GBCT source check](agents/gbct-source-and-adaptation.md).

## Data split sequence

```mermaid
flowchart TD
  n0["Split source cases or snapshots before augmentation"]
  n1["Training: fit weights and learned scales"]
  n2["Development: compare and revise recipes"]
  n3["Freeze recipe and acceptance rule"]
  n4["Independent-case test: evaluate once"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

Historical 10k campaign: 10,000 selected training states per seed from 10,010 accepted rows; 1,023 development states. Later experiments declare their training and candidate-unseen populations in [research manifests](../research/experiments/). Four training and two development snapshots come from one flame realization. Development has been inspected repeatedly. Current independent-test performance is **not evaluated**, not zero. Older CFD test results concern older models. Local-table training scores are resubstitution scores, not leave-one-out validation.

## Representation index

Here d=delta Y, Y is the initial mass fraction, and z is the encoded target before recipe-specific scaling. Encoding uses reference labels during fitting. Inference predicts z from inputs and applies the inverse; it cannot inspect the unknown reference increment.

<a id="representation-state-boxcox"></a>
### Transformed-state increment

Representation ID: `state-boxcox`.

`z = B(Y + d) - B(Y), B(y) = (y^0.1 - 1)/0.1`

Transform the states, then take their coordinate difference. Stable log1p/expm1 algebra avoids avoidable cancellation. Invalid predicted inverse domains are corrected and counted.

```mermaid
flowchart TD
  n0["Initial state Y and reference increment d"]
  n1["Stable Box-Cox state difference"]
  n2["Coordinate z"]
  n0 --> n1
  n1 --> n2
```

<a id="representation-signed-power"></a>
### Direct signed-power increment

Representation ID: `signed-power`.

`z = sign(d) * abs(d)^0.1 / 0.1`

Transform the signed physical increment directly. This is not the GBCT composition.

```mermaid
flowchart TD
  n0["Reference increment d"]
  n1["Signed power with exponent 0.1"]
  n2["Coordinate z"]
  n0 --> n1
  n1 --> n2
```

<a id="representation-budget-linear"></a>
### State-budget linear increment

Representation ID: `budget-linear`.

`z = d / (1e-12 + 1e-6 * abs(Y))`

Scale by the initial-state budget. This scale is not the increment-acceptance budget and does not use an unknown future reference at inference.

```mermaid
flowchart TD
  n0["Initial state Y"]
  n1["Compute initial-state budget"]
  n2["Divide reference increment by budget"]
  n3["Coordinate z"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="representation-scaled-asinh"></a>
### Empirical-scale asinh increment

Representation ID: `scaled-asinh`.

`z = asinh(d / s_i)`

Fit species scales s_i on training rows only. The scale is not a test-fitted parameter or a numerical accuracy guarantee.

```mermaid
flowchart TD
  n0["Training increments"]
  n1["Fit species transition scales"]
  n2["Signed asinh coordinate"]
  n3["Coordinate z"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="representation-budget-log"></a>
### Tolerance-scale signed-log increment

Representation ID: `budget-log`.

`z = sign(d) * log1p(r * abs(d) / a) / r`

Use a=1e-15 and r=0.1. The derivative is exactly 1/(a+r*abs(d)); this aligns small coordinate errors locally with the chosen increment budget.

```mermaid
flowchart TD
  n0["Reference increment d"]
  n1["Fixed tolerance scales a and r"]
  n2["Signed log1p"]
  n3["Coordinate z"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="representation-budget-asinh"></a>
### Tolerance-scale asinh increment

Representation ID: `budget-asinh`.

`z = asinh(r * d / a) / r`

Use a=1e-15 and r=0.1. The derivative is 1/sqrt(a^2+r^2*d^2), not exactly the additive acceptance rule.

```mermaid
flowchart TD
  n0["Reference increment d"]
  n1["Fixed tolerance scales a and r"]
  n2["Signed asinh"]
  n3["Coordinate z"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="representation-gbct"></a>
### GBCT transformed-state rate

Representation ID: `gbct`.

`z = sign(q) * abs(q)^0.5 / 0.5; q = [B(Y+d)-B(Y)]/h; B power = 0.1`

Matched target adaptation of GBCT, with h=1e-6 s. Stable state difference, signed square-root rate, then train-only mean/RMS normalization. Not a reproduction of the published network, dataset or loader normalization.

```mermaid
flowchart TD
  n0["Initial state Y and reference increment d"]
  n1["Stable Box-Cox difference, exponent 0.1"]
  n2["Divide by h = 1e-6 s"]
  n3["Signed power, exponent 0.5"]
  n4["Train-only mean/RMS scaling"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="representation-fixed-asinh-correction"></a>
### Fixed-crossover asinh correction

Representation ID: `fixed-asinh-correction`.

`c = asinh(d/s) - asinh(b/s); predicted d = s*sinh(asinh(b/s) + predicted c); s = 1e-14`

b is the frozen base increment. The reference-derived c is a training label, never an inference input. The crossover is fixed; it is not abs(b) or a changed accuracy allowance. Coordinate Smooth L1 trains the head; physical acceptance judges it.

```mermaid
flowchart TD
  n0["State inputs"]
  n1["Frozen base predicts increment b"]
  n2["Head predicts correction c"]
  n3["Fixed-crossover inverse gives signed increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

## Method index

Click a method name for its steps. Diagrams summarize the recipe, including fitting where labelled; they are not per-query cost diagrams.

| Stable artifact ID | Canonical name | Family |
| --- | --- | --- |
| `pool-state` | [Reaction pooling — state only](#pool-state) | Reaction-aware correction |
| `pool-affinity` | [Reaction pooling + affinity](#pool-affinity) | Reaction-aware correction |
| `attention-state` | [Reaction attention — state only](#attention-state) | Reaction-aware correction |
| `attention-affinity` | [Reaction attention + affinity](#attention-affinity) | Reaction-aware correction |
| `correction-scaling-coordinate-sl1` | [Continued-base correction — coordinate Smooth L1](#correction-scaling-coordinate-sl1) | Direct correction scaling |
| `correction-scaling-physical` | [Continued-base correction — physical supervision](#correction-scaling-physical) | Direct correction scaling |
| `correction-scaling-direct` | [Continued-base correction — direct supervision](#correction-scaling-direct) | Direct correction scaling |
| `resolved-fit-256-physical` | [Resolved small-set fit — 3x256, physical supervision](#resolved-fit-256-physical) | Training-fit diagnostic |
| `resolved-fit-256-direct` | [Resolved small-set fit — 3x256, direct supervision](#resolved-fit-256-direct) | Training-fit diagnostic |
| `resolved-fit-512-physical` | [Resolved small-set fit — 3x512, physical supervision](#resolved-fit-512-physical) | Training-fit diagnostic |
| `resolved-fit-512-direct` | [Resolved small-set fit — 3x512, direct supervision](#resolved-fit-512-direct) | Training-fit diagnostic |
| `decoder-linear-log` | [Base-relative linear correction — log loss](#decoder-linear-log) | Correction decoder and loss diagnostic |
| `decoder-linear-huber` | [Base-relative linear correction — huber loss](#decoder-linear-huber) | Correction decoder and loss diagnostic |
| `decoder-asinh-log` | [Fixed-crossover asinh correction — log loss](#decoder-asinh-log) | Correction decoder and loss diagnostic |
| `decoder-asinh-huber` | [Fixed-crossover asinh correction — huber loss](#decoder-asinh-huber) | Correction decoder and loss diagnostic |
| `adaptive-continue` | [GBCT baseline — warm-start control](#adaptive-continue) | Adaptive residual coordinates |
| `adaptive-local` | [Fixed local residual correction](#adaptive-local) | Adaptive residual coordinates |
| `adaptive-calibrated` | [Residual-calibrated local correction](#adaptive-calibrated) | Adaptive residual coordinates |
| `matched200k-state-boxcox-coordinate` | [Transformed-state increment — matched 200k coordinate loss](#matched200k-state-boxcox-coordinate) | Matched 200k targets and losses |
| `matched200k-state-boxcox-increment` | [Transformed-state increment — matched 200k increment loss](#matched200k-state-boxcox-increment) | Matched 200k targets and losses |
| `matched200k-state-boxcox-state` | [Transformed-state increment — matched 200k state loss](#matched200k-state-boxcox-state) | Matched 200k targets and losses |
| `matched200k-signed-power-coordinate` | [Direct signed-power increment — matched 200k coordinate loss](#matched200k-signed-power-coordinate) | Matched 200k targets and losses |
| `matched200k-signed-power-increment` | [Direct signed-power increment — matched 200k increment loss](#matched200k-signed-power-increment) | Matched 200k targets and losses |
| `matched200k-signed-power-state` | [Direct signed-power increment — matched 200k state loss](#matched200k-signed-power-state) | Matched 200k targets and losses |
| `matched200k-scaled-asinh-coordinate` | [Empirical-scale asinh increment — matched 200k coordinate loss](#matched200k-scaled-asinh-coordinate) | Matched 200k targets and losses |
| `matched200k-scaled-asinh-increment` | [Empirical-scale asinh increment — matched 200k increment loss](#matched200k-scaled-asinh-increment) | Matched 200k targets and losses |
| `matched200k-scaled-asinh-state` | [Empirical-scale asinh increment — matched 200k state loss](#matched200k-scaled-asinh-state) | Matched 200k targets and losses |
| `matched200k-gbct-coordinate` | [GBCT transformed-state rate — matched 200k coordinate loss](#matched200k-gbct-coordinate) | Matched 200k targets and losses |
| `matched200k-gbct-increment` | [GBCT transformed-state rate — matched 200k increment loss](#matched200k-gbct-increment) | Matched 200k targets and losses |
| `matched200k-gbct-state` | [GBCT transformed-state rate — matched 200k state loss](#matched200k-gbct-state) | Matched 200k targets and losses |
| `fuel-state` | [Transformed-state increment — Fuel source recipe](#fuel-state) | Fuel source-recipe replication |
| `fuel-power` | [Direct signed-power increment — Fuel source recipe](#fuel-power) | Fuel source-recipe replication |
| `fuel-state-matched-work` | [Transformed-state increment — matched-work control](#fuel-state-matched-work) | Matched-update data-size comparison |
| `fuel-power-matched-work` | [Direct signed-power increment — matched-work control](#fuel-power-matched-work) | Matched-update data-size comparison |
| `fuel-state-budget` | [Transformed-state increment — fixed-data budget control](#fuel-state-budget) | Fixed-data training-budget comparison |
| `fuel-power-budget` | [Direct signed-power increment — fixed-data budget control](#fuel-power-budget) | Fixed-data training-budget comparison |
| `state-boxcox-coordinate` | [Transformed-state increment — paired coordinate loss](#state-boxcox-coordinate) | Paired target and error-scale comparison |
| `state-boxcox-increment` | [Transformed-state increment — paired increment loss](#state-boxcox-increment) | Paired target and error-scale comparison |
| `state-boxcox-state` | [Transformed-state increment — paired state loss](#state-boxcox-state) | Paired target and error-scale comparison |
| `gbct-coordinate` | [GBCT transformed-state rate — paired coordinate loss](#gbct-coordinate) | Paired target and error-scale comparison |
| `gbct-increment` | [GBCT transformed-state rate — paired increment loss](#gbct-increment) | Paired target and error-scale comparison |
| `gbct-state` | [GBCT transformed-state rate — paired state loss](#gbct-state) | Paired target and error-scale comparison |
| `original-state-boxcox` | [Transformed-state increment — 2k updates](#original-state-boxcox) | Matched dense networks |
| `long-state-boxcox` | [Transformed-state increment — 4k updates](#long-state-boxcox) | Matched dense networks |
| `original-signed-power` | [Direct signed-power increment — 2k updates](#original-signed-power) | Matched dense networks |
| `long-signed-power` | [Direct signed-power increment — 4k updates](#long-signed-power) | Matched dense networks |
| `original-budget-linear` | [State-budget linear increment — 2k updates](#original-budget-linear) | Matched dense networks |
| `long-budget-linear` | [State-budget linear increment — 4k updates](#long-budget-linear) | Matched dense networks |
| `original-scaled-asinh` | [Empirical-scale asinh increment — 2k updates](#original-scaled-asinh) | Matched dense networks |
| `long-scaled-asinh` | [Empirical-scale asinh increment — 4k updates](#long-scaled-asinh) | Matched dense networks |
| `budget-log` | [Tolerance-scale signed-log increment](#budget-log) | Tolerance-derived targets |
| `physical-budget-log` | [Tolerance-scale signed-log increment + physical loss](#physical-budget-log) | Tolerance-derived targets |
| `budget-asinh` | [Tolerance-scale asinh increment](#budget-asinh) | Tolerance-derived targets |
| `physical-budget-asinh` | [Tolerance-scale asinh increment + physical loss](#physical-budget-asinh) | Tolerance-derived targets |
| `residual-state-boxcox` | [RMS-scaled additive residual](#residual-state-boxcox) | Residual corrections |
| `deep-state-boxcox` | [Transformed-state increment — 8-layer control](#deep-state-boxcox) | Capacity control |
| `continue-coordinate` | [Transformed-state coordinate continuation](#continue-coordinate) | Warm-start loss changes |
| `finetune-physical` | [Transformed-state physical fine-tuning](#finetune-physical) | Warm-start loss changes |
| `finetune-tail` | [Transformed-state tail fine-tuning](#finetune-tail) | Warm-start loss changes |
| `relative-correction` | [Base-relative asinh correction](#relative-correction) | Residual corrections |
| `protected-correction` | [Protected base-relative asinh correction](#protected-correction) | Residual corrections |
| `local-state` | [Local RBF — transformed-state increment](#local-state) | Local tables |
| `local-asinh` | [Local RBF — tolerance-scale asinh](#local-asinh) | Local tables |
| `arrhenius-heads` | [Log-partial-pressure heads — Adam](#arrhenius-heads) | Input and architecture changes |
| `arrhenius-lbfgs` | [Log-partial-pressure heads — L-BFGS/RMS](#arrhenius-lbfgs) | Input and architecture changes |
| `arrhenius-local` | [Log-partial-pressure local RBF](#arrhenius-local) | Input and architecture changes |
| `rate-euler` | [Initial-rate Euler control](#rate-euler) | Non-learned kinetics controls |
| `frozen-exponential` | [Frozen production–destruction control](#frozen-exponential) | Non-learned kinetics controls |

Report aliases: `base` → `long-state-boxcox`.

The zero-increment control always returns zero; it is not fitted.

## Method steps

<a id="pool-state"></a>
### Reaction pooling — state only

Artifact ID: `pool-state`. Family: Reaction-aware correction.

Run arm pool-state in EXP-20261014-reaction-attention. Reactant, product and collider maps form reaction tokens; fixed positive structural weights combine them for each species. No chemical-potential evaluation at inference. All species use the same declared pipeline. Stoichiometry guides aggregation, not hard output conservation. The 32-wide head has 20,449 parameters. Work and results are versioned separately; a longer run is not a new method. See research/RESULTS.md for qualification, not just development accuracy.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Frozen GBCT base and prepared state features"]
  n2["Separate reactant/product/collider reaction tokens"]
  n3["Fixed reaction pooling for each species"]
  n4["Predict fixed-asinh correction"]
  n5["Decode physical increment; score repair, damage and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="pool-affinity"></a>
### Reaction pooling + affinity

Artifact ID: `pool-affinity`. Family: Reaction-aware correction.

Run arm pool-affinity in EXP-20261014-reaction-attention. Reactant, product and collider maps form reaction tokens; fixed positive structural weights combine them for each species. Reaction affinity is minus Gibbs energy change divided by R*T, evaluated from the input state and scaled on training rows. All species use the same declared pipeline. Stoichiometry guides aggregation, not hard output conservation. The 32-wide head has 20,449 parameters. Work and results are versioned separately; a longer run is not a new method. See research/RESULTS.md for qualification, not just development accuracy.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Frozen GBCT base and prepared state features"]
  n2["Separate reactant/product/collider reaction tokens plus input-state affinity"]
  n3["Fixed reaction pooling for each species"]
  n4["Predict fixed-asinh correction"]
  n5["Decode physical increment; score repair, damage and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="attention-state"></a>
### Reaction attention — state only

Artifact ID: `attention-state`. Family: Reaction-aware correction.

Run arm attention-state in EXP-20261014-reaction-attention. Reactant, product and collider maps form reaction tokens; two-head learned attention with a structural log prior combine them for each species. No chemical-potential evaluation at inference. All species use the same declared pipeline. Stoichiometry guides aggregation, not hard output conservation. The 32-wide head has 22,561 parameters. Work and results are versioned separately; a longer run is not a new method. See research/RESULTS.md for qualification, not just development accuracy.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Frozen GBCT base and prepared state features"]
  n2["Separate reactant/product/collider reaction tokens"]
  n3["Two-head species-to-reaction attention"]
  n4["Predict fixed-asinh correction"]
  n5["Decode physical increment; score repair, damage and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="attention-affinity"></a>
### Reaction attention + affinity

Artifact ID: `attention-affinity`. Family: Reaction-aware correction.

Run arm attention-affinity in EXP-20261014-reaction-attention. Reactant, product and collider maps form reaction tokens; two-head learned attention with a structural log prior combine them for each species. Reaction affinity is minus Gibbs energy change divided by R*T, evaluated from the input state and scaled on training rows. All species use the same declared pipeline. Stoichiometry guides aggregation, not hard output conservation. The 32-wide head has 22,561 parameters. Work and results are versioned separately; a longer run is not a new method. See research/RESULTS.md for qualification, not just development accuracy.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Frozen GBCT base and prepared state features"]
  n2["Separate reactant/product/collider reaction tokens plus input-state affinity"]
  n3["Two-head species-to-reaction attention"]
  n4["Predict fixed-asinh correction"]
  n5["Decode physical increment; score repair, damage and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="correction-scaling-coordinate-sl1"></a>
### Continued-base correction — coordinate Smooth L1

Artifact ID: `correction-scaling-coordinate-sl1`. Family: Direct correction scaling.

Bounded-influence coordinate supervision, beta=log(1.1). Same frozen continued bases, 10k rows, features, zero-output 3x256 head, fixed asinh crossover 1e-14, 6k updates and batch 10k as the completed direct-MSE controls. No physical decoder in the loss gradient path. Linear loss tail, not a hard cap. No preservation penalty, balancing, capacity or label change. Training-tail inspection and a training-mean constant control precede fitting. Physical acceptance, repair/damage and conditional tails decide success; lower coordinate loss alone does not.

```mermaid
flowchart TD
  n0["Freeze continued GBCT and existing prepared 10k inputs"]
  n1["Inspect correction tail and constant-shift control"]
  n2["Predict correction coordinate with paired fresh 3x256 head"]
  n3["Train coordinate Smooth L1; beta = log(1.1)"]
  n4["Same stable asinh decoder"]
  n5["Require physical repair with preservation; no automatic size increase"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="correction-scaling-physical"></a>
### Continued-base correction — physical supervision

Artifact ID: `correction-scaling-physical`. Family: Direct correction scaling.

Freeze continued GBCT incumbents, not the original bases. Fixed 50k input/base-feature scaling and immutable targets. Fresh zero-output 3x256 GELU, fixed asinh crossover 1e-14, direct correction MSE versus physical log1p error. Nested 10k/50k/200k correction-training prefixes, conditional progression, 6k updates and batch 10k. Same acceptance rule. No balancing, extra capacity, CFD or independent test. Base already saw the full training pool; not a from-scratch data requirement.

```mermaid
flowchart TD
  n0["Freeze continued GBCT base and audit coverage"]
  n1["Prepare training-only base features and correction coordinates"]
  n2["Fresh 3x256 head on selected nested prefix"]
  n3["6000 matched updates with physical supervision"]
  n4["Stable FP64 inverse; train/development group acceptance and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="correction-scaling-direct"></a>
### Continued-base correction — direct supervision

Artifact ID: `correction-scaling-direct`. Family: Direct correction scaling.

Freeze continued GBCT incumbents, not the original bases. Fixed 50k input/base-feature scaling and immutable targets. Fresh zero-output 3x256 GELU, fixed asinh crossover 1e-14, direct correction MSE versus physical log1p error. Nested 10k/50k/200k correction-training prefixes, conditional progression, 6k updates and batch 10k. Same acceptance rule. No balancing, extra capacity, CFD or independent test. Base already saw the full training pool; not a from-scratch data requirement.

```mermaid
flowchart TD
  n0["Freeze continued GBCT base and audit coverage"]
  n1["Prepare training-only base features and correction coordinates"]
  n2["Fresh 3x256 head on selected nested prefix"]
  n3["6000 matched updates with direct supervision"]
  n4["Stable FP64 inverse; train/development group acceptance and cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="resolved-fit-256-physical"></a>
### Resolved small-set fit — 3x256, physical supervision

Artifact ID: `resolved-fit-256-physical`. Family: Training-fit diagnostic.

All 16 audited training states; same resolved-component mask and audit reference increments. Frozen GBCT base, fixed asinh crossover 1e-14, original 119 features, zero-output correction head. Direct correction-coordinate MSE versus physical log1p normalized error. Two seeds, 6k full-batch updates, Adam cosine 1e-4 to 1e-6, final checkpoint. Equal updates, not equal compute. Training-fit evidence only: no development or test score and no claim of generalization.

```mermaid
flowchart TD
  n0["Existing independently audited training rows"]
  n1["Mask unresolved species components"]
  n2["Frozen GBCT base plus fixed-asinh correction"]
  n3["3x256 head; physical supervision"]
  n4["Training-only acceptance, original failures and GPU cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="resolved-fit-256-direct"></a>
### Resolved small-set fit — 3x256, direct supervision

Artifact ID: `resolved-fit-256-direct`. Family: Training-fit diagnostic.

All 16 audited training states; same resolved-component mask and audit reference increments. Frozen GBCT base, fixed asinh crossover 1e-14, original 119 features, zero-output correction head. Direct correction-coordinate MSE versus physical log1p normalized error. Two seeds, 6k full-batch updates, Adam cosine 1e-4 to 1e-6, final checkpoint. Equal updates, not equal compute. Training-fit evidence only: no development or test score and no claim of generalization.

```mermaid
flowchart TD
  n0["Existing independently audited training rows"]
  n1["Mask unresolved species components"]
  n2["Frozen GBCT base plus fixed-asinh correction"]
  n3["3x256 head; direct supervision"]
  n4["Training-only acceptance, original failures and GPU cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="resolved-fit-512-physical"></a>
### Resolved small-set fit — 3x512, physical supervision

Artifact ID: `resolved-fit-512-physical`. Family: Training-fit diagnostic.

All 16 audited training states; same resolved-component mask and audit reference increments. Frozen GBCT base, fixed asinh crossover 1e-14, original 119 features, zero-output correction head. Direct correction-coordinate MSE versus physical log1p normalized error. Two seeds, 6k full-batch updates, Adam cosine 1e-4 to 1e-6, final checkpoint. Equal updates, not equal compute. Training-fit evidence only: no development or test score and no claim of generalization.

```mermaid
flowchart TD
  n0["Existing independently audited training rows"]
  n1["Mask unresolved species components"]
  n2["Frozen GBCT base plus fixed-asinh correction"]
  n3["3x512 head; physical supervision"]
  n4["Training-only acceptance, original failures and GPU cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="resolved-fit-512-direct"></a>
### Resolved small-set fit — 3x512, direct supervision

Artifact ID: `resolved-fit-512-direct`. Family: Training-fit diagnostic.

All 16 audited training states; same resolved-component mask and audit reference increments. Frozen GBCT base, fixed asinh crossover 1e-14, original 119 features, zero-output correction head. Direct correction-coordinate MSE versus physical log1p normalized error. Two seeds, 6k full-batch updates, Adam cosine 1e-4 to 1e-6, final checkpoint. Equal updates, not equal compute. Training-fit evidence only: no development or test score and no claim of generalization.

```mermaid
flowchart TD
  n0["Existing independently audited training rows"]
  n1["Mask unresolved species components"]
  n2["Frozen GBCT base plus fixed-asinh correction"]
  n3["3x512 head; direct supervision"]
  n4["Training-only acceptance, original failures and GPU cost"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="decoder-linear-log"></a>
### Base-relative linear correction — log loss

Artifact ID: `decoder-linear-log`. Family: Correction decoder and loss diagnostic.

Same frozen 200k GBCT/increment base, 3x256 GELU zero-output head, input plus base-output features, two seeds, 6k extra updates and batch 10k. Alpha=1, no coordinate normalization. Linear decoder b+(1e-14+abs(b))*c; asinh decoder 1e-14*sinh(asinh(b/1e-14)+c), with stable zero-identity arithmetic. Log1p absolute normalized error versus Huber on signed normalized error (delta=1). Fixed allowance 1e-15+.1*abs(reference). No gradient clipping; record raw norms. No-training FP32 oracle audit and small-target overprediction counts. Not a demonstrated gain or a new tolerance rule.

```mermaid
flowchart TD
  n0["Frozen GBCT predicts base b"]
  n1["Identical inputs and zero-initialized correction head"]
  n2["Decode using linear; FP64 reconstruction"]
  n3["Train with normalized log error"]
  n4["Same increment acceptance, conditional tails and sign checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="decoder-linear-huber"></a>
### Base-relative linear correction — huber loss

Artifact ID: `decoder-linear-huber`. Family: Correction decoder and loss diagnostic.

Same frozen 200k GBCT/increment base, 3x256 GELU zero-output head, input plus base-output features, two seeds, 6k extra updates and batch 10k. Alpha=1, no coordinate normalization. Linear decoder b+(1e-14+abs(b))*c; asinh decoder 1e-14*sinh(asinh(b/1e-14)+c), with stable zero-identity arithmetic. Log1p absolute normalized error versus Huber on signed normalized error (delta=1). Fixed allowance 1e-15+.1*abs(reference). No gradient clipping; record raw norms. No-training FP32 oracle audit and small-target overprediction counts. Not a demonstrated gain or a new tolerance rule.

```mermaid
flowchart TD
  n0["Frozen GBCT predicts base b"]
  n1["Identical inputs and zero-initialized correction head"]
  n2["Decode using linear; FP64 reconstruction"]
  n3["Train with normalized huber error"]
  n4["Same increment acceptance, conditional tails and sign checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="decoder-asinh-log"></a>
### Fixed-crossover asinh correction — log loss

Artifact ID: `decoder-asinh-log`. Family: Correction decoder and loss diagnostic.

Same frozen 200k GBCT/increment base, 3x256 GELU zero-output head, input plus base-output features, two seeds, 6k extra updates and batch 10k. Alpha=1, no coordinate normalization. Linear decoder b+(1e-14+abs(b))*c; asinh decoder 1e-14*sinh(asinh(b/1e-14)+c), with stable zero-identity arithmetic. Log1p absolute normalized error versus Huber on signed normalized error (delta=1). Fixed allowance 1e-15+.1*abs(reference). No gradient clipping; record raw norms. No-training FP32 oracle audit and small-target overprediction counts. Not a demonstrated gain or a new tolerance rule.

```mermaid
flowchart TD
  n0["Frozen GBCT predicts base b"]
  n1["Identical inputs and zero-initialized correction head"]
  n2["Decode using asinh; FP64 reconstruction"]
  n3["Train with normalized log error"]
  n4["Same increment acceptance, conditional tails and sign checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="decoder-asinh-huber"></a>
### Fixed-crossover asinh correction — huber loss

Artifact ID: `decoder-asinh-huber`. Family: Correction decoder and loss diagnostic.

Same frozen 200k GBCT/increment base, 3x256 GELU zero-output head, input plus base-output features, two seeds, 6k extra updates and batch 10k. Alpha=1, no coordinate normalization. Linear decoder b+(1e-14+abs(b))*c; asinh decoder 1e-14*sinh(asinh(b/1e-14)+c), with stable zero-identity arithmetic. Log1p absolute normalized error versus Huber on signed normalized error (delta=1). Fixed allowance 1e-15+.1*abs(reference). No gradient clipping; record raw norms. No-training FP32 oracle audit and small-target overprediction counts. Not a demonstrated gain or a new tolerance rule.

```mermaid
flowchart TD
  n0["Frozen GBCT predicts base b"]
  n1["Identical inputs and zero-initialized correction head"]
  n2["Decode using asinh; FP64 reconstruction"]
  n3["Train with normalized huber error"]
  n4["Same increment acceptance, conditional tails and sign checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="adaptive-continue"></a>
### GBCT baseline — warm-start control

Artifact ID: `adaptive-continue`. Family: Adaptive residual coordinates.

Frozen 200k dataset and verified GBCT/increment base. Two seeds, 6k extra updates, batch 10k, fresh Adam, cosine 1e-4 to 1e-6, final checkpoint. Same physical increment loss and both evaluation policies. Continuation fits the base; corrections freeze it and add a zero-output 3x256 GELU model with normalized physical and asinh(base) features. Decoder: base+(1e-14+abs(base))*alpha(base)*correction. Local alpha=1; calibrated alpha uses bounded conditional training-residual 75th percentiles in eight fixed magnitude bins, fit on 50k training rows only. No positivity projection or test access. Scale is not an acceptance tolerance. Experimental, not a demonstrated improvement or MSNN reproduction.

```mermaid
flowchart TD
  n0["Verified 200k GBCT/increment model"]
  n1["Fresh Adam; 6k physical-loss updates"]
  n2["Same physical inverse and acceptance rules"]
  n0 --> n1
  n1 --> n2
```

<a id="adaptive-local"></a>
### Fixed local residual correction

Artifact ID: `adaptive-local`. Family: Adaptive residual coordinates.

Frozen 200k dataset and verified GBCT/increment base. Two seeds, 6k extra updates, batch 10k, fresh Adam, cosine 1e-4 to 1e-6, final checkpoint. Same physical increment loss and both evaluation policies. Continuation fits the base; corrections freeze it and add a zero-output 3x256 GELU model with normalized physical and asinh(base) features. Decoder: base+(1e-14+abs(base))*alpha(base)*correction. Local alpha=1; calibrated alpha uses bounded conditional training-residual 75th percentiles in eight fixed magnitude bins, fit on 50k training rows only. No positivity projection or test access. Scale is not an acceptance tolerance. Experimental, not a demonstrated improvement or MSNN reproduction.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Frozen GBCT/increment model predicts base"]
  n2["Local scale 1e-14 + abs(base)"]
  n3["Zero-initialized 3x256 correction network"]
  n4["FP64 base + scaled correction; no projection"]
  n5["Unchanged acceptance rules and physical checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="adaptive-calibrated"></a>
### Residual-calibrated local correction

Artifact ID: `adaptive-calibrated`. Family: Adaptive residual coordinates.

Frozen 200k dataset and verified GBCT/increment base. Two seeds, 6k extra updates, batch 10k, fresh Adam, cosine 1e-4 to 1e-6, final checkpoint. Same physical increment loss and both evaluation policies. Continuation fits the base; corrections freeze it and add a zero-output 3x256 GELU model with normalized physical and asinh(base) features. Decoder: base+(1e-14+abs(base))*alpha(base)*correction. Local alpha=1; calibrated alpha uses bounded conditional training-residual 75th percentiles in eight fixed magnitude bins, fit on 50k training rows only. No positivity projection or test access. Scale is not an acceptance tolerance. Experimental, not a demonstrated improvement or MSNN reproduction.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Frozen GBCT/increment model predicts base"]
  n2["Local scale times interpolated training-residual calibration"]
  n3["Zero-initialized 3x256 correction network"]
  n4["FP64 base + scaled correction; no projection"]
  n5["Unchanged acceptance rules and physical checks"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-state-boxcox-coordinate"></a>
### Transformed-state increment — matched 200k coordinate loss

Artifact ID: `matched200k-state-boxcox-coordinate`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Transformed-state increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k coordinate-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-state-boxcox-increment"></a>
### Transformed-state increment — matched 200k increment loss

Artifact ID: `matched200k-state-boxcox-increment`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Transformed-state increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k increment-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-state-boxcox-state"></a>
### Transformed-state increment — matched 200k state loss

Artifact ID: `matched200k-state-boxcox-state`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Transformed-state increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k state-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-signed-power-coordinate"></a>
### Direct signed-power increment — matched 200k coordinate loss

Artifact ID: `matched200k-signed-power-coordinate`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Direct signed-power increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k coordinate-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-signed-power-increment"></a>
### Direct signed-power increment — matched 200k increment loss

Artifact ID: `matched200k-signed-power-increment`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Direct signed-power increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k increment-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-signed-power-state"></a>
### Direct signed-power increment — matched 200k state loss

Artifact ID: `matched200k-signed-power-state`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Direct signed-power increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k state-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-scaled-asinh-coordinate"></a>
### Empirical-scale asinh increment — matched 200k coordinate loss

Artifact ID: `matched200k-scaled-asinh-coordinate`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Empirical-scale asinh increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k coordinate-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-scaled-asinh-increment"></a>
### Empirical-scale asinh increment — matched 200k increment loss

Artifact ID: `matched200k-scaled-asinh-increment`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Empirical-scale asinh increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k increment-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-scaled-asinh-state"></a>
### Empirical-scale asinh increment — matched 200k state loss

Artifact ID: `matched200k-scaled-asinh-state`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict Empirical-scale asinh increment"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k state-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-gbct-coordinate"></a>
### GBCT transformed-state rate — matched 200k coordinate loss

Artifact ID: `matched200k-gbct-coordinate`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict GBCT transformed-state rate"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k coordinate-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-gbct-increment"></a>
### GBCT transformed-state rate — matched 200k increment loss

Artifact ID: `matched200k-gbct-increment`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict GBCT transformed-state rate"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k increment-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="matched200k-gbct-state"></a>
### GBCT transformed-state rate — matched 200k state loss

Artifact ID: `matched200k-gbct-state`. Family: Matched 200k targets and losses.

Fixed 200k rows, mean/sample-standard-deviation scalers fitted on the same 50k prefix. Fresh 4x800 GELU, 58 non-AR outputs, batch 10k. Common 18k schedule: 6k each at 1e-3/1e-4/1e-5, with Adam reset each stage. First 12k coordinate L1, last 6k selected loss. Physical loss is mean log1p(error/allowance), a=1e-15 and r=.1; increment scale abs(d), state scale abs(Y+d). Asinh scale 1e-14; GBCT a=.1,b=.5,h=1e-6. FP32 fit/FP64 inverse, final checkpoint. Not an original paper recipe.

```mermaid
flowchart TD
  n0["Same 200k training rows and seed"]
  n1["Freeze common input and target scaling on 50k prefix"]
  n2["Predict GBCT transformed-state rate"]
  n3["12k identical coordinate-loss warmup within each target"]
  n4["6k state-loss updates under common schedule"]
  n5["FP64 physical reconstruction; score BOTH error scales and tails"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-state"></a>
### Transformed-state increment — Fuel source recipe

Artifact ID: `fuel-state`. Family: Fuel source-recipe replication.

4x800 GELU, 58 non-argon outputs, FP32 L1. Train-only centered sample-standard-deviation scaling. 1500 shuffled epochs; requested batch 20k capped at training count. Adam resets before epochs 502 and 1002 with tenfold LR drops. Stable FP64 inverse. Reduced-data diagnostic, not full paper reproduction.

```mermaid
flowchart TD
  n0["Same checked training states"]
  n1["Sample-standard-deviation input and target scaling"]
  n2["61 to 800x4 to 58 GELU network"]
  n3["1500 epochs; source Adam resets"]
  n4["Stable transformed-state inverse"]
  n5["Both error budgets and SSPI"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-power"></a>
### Direct signed-power increment — Fuel source recipe

Artifact ID: `fuel-power`. Family: Fuel source-recipe replication.

4x800 GELU, 58 non-argon outputs, FP32 L1. Train-only population-standard-deviation scaling; zero target center, not RMS. 2000 shuffled epochs; requested batch 20k capped at training count. Tenfold LR drop every 500 epochs; retain Adam state. Stable FP64 inverse. Reduced-data diagnostic, not full paper reproduction.

```mermaid
flowchart TD
  n0["Same checked training states"]
  n1["Signed tenth-root increment; zero target center"]
  n2["Population-standard-deviation scaling"]
  n3["61 to 800x4 to 58 GELU network"]
  n4["2000 epochs; StepLR with Adam state retained"]
  n5["Direct power inverse; both budgets and SSPI"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-state-matched-work"></a>
### Transformed-state increment — matched-work control

Artifact ID: `fuel-state-matched-work`. Family: Matched-update data-size comparison.

50k/200k nested rows, fixed 50k training-only scalers. Same seeded 4x800 GELU, FP32 L1, 6000 updates, batch 10k, 60M presentations. Three 2000-update LR stages; reset Adam. Final checkpoint, stable FP64 inverse. Not the original epoch schedule; compare sizes within each recipe.

```mermaid
flowchart TD
  n0["Nested 50k or 200k training rows"]
  n1["Freeze input and target scaling fitted on 50k"]
  n2["Same seeded 61 to 800x4 to 58 GELU network"]
  n3["6000 updates; 10k rows per update"]
  n4["Stable physical reconstruction"]
  n5["Same development states; both error policies"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-power-matched-work"></a>
### Direct signed-power increment — matched-work control

Artifact ID: `fuel-power-matched-work`. Family: Matched-update data-size comparison.

50k/200k nested rows, fixed 50k training-only scalers. Same seeded 4x800 GELU, FP32 L1, 6000 updates, batch 10k, 60M presentations. Four 1500-update LR stages; retain Adam. Final checkpoint, stable FP64 inverse. Not the original epoch schedule; compare sizes within each recipe.

```mermaid
flowchart TD
  n0["Nested 50k or 200k training rows"]
  n1["Freeze input and target scaling fitted on 50k"]
  n2["Same seeded 61 to 800x4 to 58 GELU network"]
  n3["6000 updates; 10k rows per update"]
  n4["Stable physical reconstruction"]
  n5["Same development states; both error policies"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-state-budget"></a>
### Transformed-state increment — fixed-data budget control

Artifact ID: `fuel-state-budget`. Family: Fixed-data training-budget comparison.

Same 200k rows, original 50k scalers, seed and first 6k batches. Fresh 6k/18k update fits, batch 10k, 60M/180M presentations. Stretch source-derived learning-rate stages threefold; keep recipe Adam policy. Same 4x800 GELU, FP32 L1 and FP64 inverse. Final checkpoint only; not continuation or pure schedule-isolated effect.

```mermaid
flowchart TD
  n0["Same checked 200k training pool"]
  n1["Freeze original 50k preprocessing"]
  n2["Same seeded 61 to 800x4 to 58 network"]
  n3["Fresh 6k or 18k update fit; proportional LR stages"]
  n4["Stable FP64 physical reconstruction"]
  n5["Same development rows; both error scales and zero control"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="fuel-power-budget"></a>
### Direct signed-power increment — fixed-data budget control

Artifact ID: `fuel-power-budget`. Family: Fixed-data training-budget comparison.

Same 200k rows, original 50k scalers, seed and first 6k batches. Fresh 6k/18k update fits, batch 10k, 60M/180M presentations. Stretch source-derived learning-rate stages threefold; keep recipe Adam policy. Same 4x800 GELU, FP32 L1 and FP64 inverse. Final checkpoint only; not continuation or pure schedule-isolated effect.

```mermaid
flowchart TD
  n0["Same checked 200k training pool"]
  n1["Freeze original 50k preprocessing"]
  n2["Same seeded 61 to 800x4 to 58 network"]
  n3["Fresh 6k or 18k update fit; proportional LR stages"]
  n4["Stable FP64 physical reconstruction"]
  n5["Same development rows; both error scales and zero control"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="state-boxcox-coordinate"></a>
### Transformed-state increment — paired coordinate loss

Artifact ID: `state-boxcox-coordinate`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k coordinate updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict Transformed-state increment"]
  n2["2k coordinate warmup"]
  n3["2k coordinate-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="state-boxcox-increment"></a>
### Transformed-state increment — paired increment loss

Artifact ID: `state-boxcox-increment`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k increment updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict Transformed-state increment"]
  n2["2k coordinate warmup"]
  n3["2k increment-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="state-boxcox-state"></a>
### Transformed-state increment — paired state loss

Artifact ID: `state-boxcox-state`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k state updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict Transformed-state increment"]
  n2["2k coordinate warmup"]
  n3["2k state-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="gbct-coordinate"></a>
### GBCT transformed-state rate — paired coordinate loss

Artifact ID: `gbct-coordinate`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k coordinate updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict GBCT transformed-state rate"]
  n2["2k coordinate warmup"]
  n3["2k coordinate-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="gbct-increment"></a>
### GBCT transformed-state rate — paired increment loss

Artifact ID: `gbct-increment`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k increment updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict GBCT transformed-state rate"]
  n2["2k coordinate warmup"]
  n3["2k increment-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="gbct-state"></a>
### GBCT transformed-state rate — paired state loss

Artifact ID: `gbct-state`. Family: Paired target and error-scale comparison.

Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k state updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.

```mermaid
flowchart TD
  n0["Same 10k training states and seed"]
  n1["Predict GBCT transformed-state rate"]
  n2["2k coordinate warmup"]
  n3["2k state-loss updates"]
  n4["Stable physical increment inverse"]
  n5["Score BOTH increment and state allowances"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="original-state-boxcox"></a>
### Transformed-state increment — 2k updates

Artifact ID: `original-state-boxcox`. Family: Matched dense networks.

4x800 GELU, FP32, L1 coordinate loss, Adam, 2,000 updates. Training-only output standardization. State count and seed belong to the run.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Transformed-state increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="long-state-boxcox"></a>
### Transformed-state increment — 4k updates

Artifact ID: `long-state-boxcox`. Family: Matched dense networks.

Same 4x800 recipe; fresh 4,000-update cosine schedule from the original initialization, not continued Adam state. 10,000 selected training rows.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Transformed-state increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="original-signed-power"></a>
### Direct signed-power increment — 2k updates

Artifact ID: `original-signed-power`. Family: Matched dense networks.

4x800 GELU, FP32, L1 coordinate loss, Adam, 2,000 updates. Training-only output standardization. State count and seed belong to the run.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Direct signed-power increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="long-signed-power"></a>
### Direct signed-power increment — 4k updates

Artifact ID: `long-signed-power`. Family: Matched dense networks.

Same 4x800 recipe; fresh 4,000-update cosine schedule from the original initialization, not continued Adam state. 10,000 selected training rows.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Direct signed-power increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="original-budget-linear"></a>
### State-budget linear increment — 2k updates

Artifact ID: `original-budget-linear`. Family: Matched dense networks.

4x800 GELU, FP32, L1 coordinate loss, Adam, 2,000 updates. Training-only output standardization. State count and seed belong to the run.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts State-budget linear increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="long-budget-linear"></a>
### State-budget linear increment — 4k updates

Artifact ID: `long-budget-linear`. Family: Matched dense networks.

Same 4x800 recipe; fresh 4,000-update cosine schedule from the original initialization, not continued Adam state. 10,000 selected training rows.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts State-budget linear increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="original-scaled-asinh"></a>
### Empirical-scale asinh increment — 2k updates

Artifact ID: `original-scaled-asinh`. Family: Matched dense networks.

4x800 GELU, FP32, L1 coordinate loss, Adam, 2,000 updates. Training-only output standardization. State count and seed belong to the run.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Empirical-scale asinh increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="long-scaled-asinh"></a>
### Empirical-scale asinh increment — 4k updates

Artifact ID: `long-scaled-asinh`. Family: Matched dense networks.

Same 4x800 recipe; fresh 4,000-update cosine schedule from the original initialization, not continued Adam state. 10,000 selected training rows.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Empirical-scale asinh increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="budget-log"></a>
### Tolerance-scale signed-log increment

Artifact ID: `budget-log`. Family: Tolerance-derived targets.

4x800, 2,000 updates, L1 coordinate loss. Fixed coordinate divisor 100 and zero offset; no species-wise output standardization.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Tolerance-scale signed-log increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="physical-budget-log"></a>
### Tolerance-scale signed-log increment + physical loss

Artifact ID: `physical-budget-log`. Family: Tolerance-derived targets.

Same recipe plus 0.01 times mean log1p(abs(physical error)/budget). This is a target-and-loss method, not a new representation.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Tolerance-scale signed-log increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="budget-asinh"></a>
### Tolerance-scale asinh increment

Artifact ID: `budget-asinh`. Family: Tolerance-derived targets.

4x800, 2,000 updates, L1 coordinate loss. Fixed coordinate divisor 100 and zero offset; no species-wise output standardization.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Tolerance-scale asinh increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="physical-budget-asinh"></a>
### Tolerance-scale asinh increment + physical loss

Artifact ID: `physical-budget-asinh`. Family: Tolerance-derived targets.

Same recipe plus 0.01 times mean log1p(abs(physical error)/budget). This is a target-and-loss method, not a new representation.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts Tolerance-scale asinh increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="residual-state-boxcox"></a>
### RMS-scaled additive residual

Artifact ID: `residual-state-boxcox`. Family: Residual corrections.

Freeze the original 2k base. Fit one 4x800 network for 2k updates to FP64 physical residuals, divided by per-species training RMS. Reconstruct and add in FP64. This is not the later base-relative correction.

```mermaid
flowchart TD
  n0["Frozen 2k base predicts d0"]
  n1["Training residual = reference minus d0"]
  n2["Training-only species RMS scale"]
  n3["4x800 network predicts normalized residual"]
  n4["FP64 d0 plus predicted residual"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="deep-state-boxcox"></a>
### Transformed-state increment — 8-layer control

Artifact ID: `deep-state-boxcox`. Family: Capacity control.

8x800 GELU, 2,000 updates. Approximate parameter/work control for two networks; measured CPU time is not forced equal.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Training-only input scaling"]
  n2["Dense network predicts transformed-state increment"]
  n3["Undo target scaling and coordinate transform"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="continue-coordinate"></a>
### Transformed-state coordinate continuation

Artifact ID: `continue-coordinate`. Family: Warm-start loss changes.

4,000 additional updates on the verified 4k base with its coordinate loss; 8k total updates.

```mermaid
flowchart TD
  n0["Verified 4k transformed-state network"]
  n1["4k further updates: coordinate loss"]
  n2["Same transformed-state inverse"]
  n3["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="finetune-physical"></a>
### Transformed-state physical fine-tuning

Artifact ID: `finetune-physical`. Family: Warm-start loss changes.

4,000 additional updates, mean log1p(abs(physical error)/budget). Output representation stays unchanged.

```mermaid
flowchart TD
  n0["Verified 4k transformed-state network"]
  n1["4k further updates: physical loss"]
  n2["Same transformed-state inverse"]
  n3["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="finetune-tail"></a>
### Transformed-state tail fine-tuning

Artifact ID: `finetune-tail`. Family: Warm-start loss changes.

Physical fine-tuning plus 0.25 times the mean worst-eight-species loss per state. Not the ADA-CVaR sampling algorithm.

```mermaid
flowchart TD
  n0["Verified 4k transformed-state network"]
  n1["4k further updates: physical plus tail loss"]
  n2["Same transformed-state inverse"]
  n3["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="relative-correction"></a>
### Base-relative asinh correction

Artifact ID: `relative-correction`. Family: Residual corrections.

Frozen 4k base, zero-initialized 4x256 correction, 4k updates. Scale = 1e-14 + abs(base prediction), available at inference. It does not use reference-dependent decoding.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Frozen 4k base predicts d0"]
  n2["4x256 network predicts local-asinh correction"]
  n3["Scale from 1e-14 + abs(d0), not reference"]
  n4["Stable inverse gives corrected increment"]
  n5["Training uses physical normalized error"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="protected-correction"></a>
### Protected base-relative asinh correction

Artifact ID: `protected-correction`. Family: Residual corrections.

Same correction plus tail loss and training-only penalty for damaging previously passing components. This does not guarantee non-regression on new states.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Frozen 4k base predicts d0"]
  n2["4x256 network predicts local-asinh correction"]
  n3["Scale from 1e-14 + abs(d0), not reference"]
  n4["Stable inverse gives corrected increment"]
  n5["Training adds tail and non-regression penalties"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="local-state"></a>
### Local RBF — transformed-state increment

Artifact ID: `local-state`. Family: Local tables.

128-neighbor cubic radial-basis interpolation, degree-one polynomial, smoothing 1e-8. Training-only numerical-rank basis. Not ISAT or certified tabulation.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Training-only scaling and rank reduction"]
  n2["128 nearest training neighbors"]
  n3["Cubic RBF predicts transformed-state increment"]
  n4["Inverse coordinate gives increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="local-asinh"></a>
### Local RBF — tolerance-scale asinh

Artifact ID: `local-asinh`. Family: Local tables.

Same local table with asinh(d/1e-14) coordinates. Fixed multiplicative coordinate factors do not change the reconstructed interpolant. Query cost includes the local solve.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Training-only scaling and rank reduction"]
  n2["128 nearest training neighbors"]
  n3["Cubic RBF predicts tolerance-scale asinh"]
  n4["Inverse coordinate gives increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="arrhenius-heads"></a>
### Log-partial-pressure heads — Adam

Artifact ID: `arrhenius-heads`. Family: Input and architecture changes.

Separate 2x32 tanh species heads. 4k coordinate updates then 4k physical updates. Standardized asinh(d/1e-14). The historical arrhenius ID denotes motivated inputs, not an exact Arrhenius law.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Inverse T, log pressure, log-like partial pressures"]
  n2["Separate 2x32 tanh network per species"]
  n3["4k coordinate warmup"]
  n4["4k Adam physical-loss updates"]
  n5["Inverse asinh gives increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="arrhenius-lbfgs"></a>
### Log-partial-pressure heads — L-BFGS/RMS

Artifact ID: `arrhenius-lbfgs`. Family: Input and architecture changes.

Same coordinate warmup, then 80 full-batch L-BFGS steps, maximum 400 closure evaluations, species-RMS physical objective. Changes optimizer and loss together.

```mermaid
flowchart TD
  n0["T, pressure, composition"]
  n1["Inverse T, log pressure, log-like partial pressures"]
  n2["Separate 2x32 tanh network per species"]
  n3["4k coordinate warmup"]
  n4["80 full-batch L-BFGS/RMS steps"]
  n5["Inverse asinh gives increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
  n4 --> n5
```

<a id="arrhenius-local"></a>
### Log-partial-pressure local RBF

Artifact ID: `arrhenius-local`. Family: Input and architecture changes.

New inverse-T/log-partial-pressure inputs with the same local asinh interpolation. It has no neural training or base predictor.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Inverse T and log-like partial pressures"]
  n2["Training-only scaling and rank reduction"]
  n3["128-neighbor asinh RBF"]
  n4["Inverse asinh gives increment"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

<a id="rate-euler"></a>
### Initial-rate Euler control

Artifact ID: `rate-euler`. Family: Non-learned kinetics controls.

Actual mechanism evaluation at inference. One explicit fixed step, no learned weights and no adaptive error control.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Mechanism gives initial net species rates"]
  n2["Multiply rates by interval"]
  n3["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
```

<a id="frozen-exponential"></a>
### Frozen production–destruction control

Artifact ID: `frozen-exponential`. Family: Non-learned kinetics controls.

Freeze species production and destruction coefficients during the interval; use stable expm1 reconstruction. Positivity does not imply conservation or accuracy.

```mermaid
flowchart TD
  n0["Input state"]
  n1["Mechanism gives production and destruction"]
  n2["Freeze each species coefficient"]
  n3["Analytic scalar exponential step"]
  n4["Physical increment prediction"]
  n0 --> n1
  n1 --> n2
  n2 --> n3
  n3 --> n4
```

## Executable sources

- [Original coordinates](../benchmarks/flame_conditioning/coordinates.py) and [learning settings](../benchmarks/offline_accuracy/learning.json).
- [Refinement plan](../benchmarks/offline_accuracy/refinement/plan.py) and [coordinates](../benchmarks/offline_accuracy/refinement/coordinates.py).
- [Adaptation plan](../benchmarks/offline_accuracy/improve/plan.py), [neural fitting](../benchmarks/offline_accuracy/improve/neural.py), [local fitting](../benchmarks/offline_accuracy/improve/local.py), and [input/head adaptation](../benchmarks/offline_accuracy/improve/arrhenius.py).
- [Non-learned controls](../benchmarks/offline_accuracy/improve/physics_prior.py).

- [GBCT and paired-scaling plan](../benchmarks/offline_accuracy/paired/README.md), [configuration](../benchmarks/offline_accuracy/paired/plan.py), and [stable coordinates](../benchmarks/offline_accuracy/paired/coordinates.py).

- [Fuel source-recipe replication](../benchmarks/offline_accuracy/paper_baseline/README.md) and [original-source audit](agents/fuel-baseline-replication.md).

- [Reaction structure, affinity and attention](../benchmarks/offline_accuracy/paper_baseline/reaction_attention/README.md) and [current physical results](../research/RESULTS.md).

Executable plans and saved run configurations own numerical parameters. This registry owns review names, diagrams and alias mappings. A display-name change never rewrites a run artifact.
