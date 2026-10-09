// Single source of display names and method diagrams. Artifact IDs stay unchanged.
// Recipe parameters remain executable in the experiment plans; this is their review index.
export const catalogueUrl = "https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/docs/research-method-catalogue.md";

export const representations = [
  {id:"state-boxcox", name:"Transformed-state increment", formula:"z = B(Y + d) - B(Y), B(y) = (y^0.1 - 1)/0.1",
    detail:"Transform the states, then take their coordinate difference. Stable log1p/expm1 algebra avoids avoidable cancellation. Invalid predicted inverse domains are corrected and counted.",
    flow:["Initial state Y and reference increment d", "Stable Box-Cox state difference", "Coordinate z"]},
  {id:"signed-power", name:"Direct signed-power increment", formula:"z = sign(d) * abs(d)^0.1 / 0.1",
    detail:"Transform the signed physical increment directly. This is not the GBCT composition.",
    flow:["Reference increment d", "Signed power with exponent 0.1", "Coordinate z"]},
  {id:"budget-linear", name:"State-budget linear increment", formula:"z = d / (1e-12 + 1e-6 * abs(Y))",
    detail:"Scale by the initial-state budget. This scale is not the increment-acceptance budget and does not use an unknown future reference at inference.",
    flow:["Initial state Y", "Compute initial-state budget", "Divide reference increment by budget", "Coordinate z"]},
  {id:"scaled-asinh", name:"Empirical-scale asinh increment", formula:"z = asinh(d / s_i)",
    detail:"Fit species scales s_i on training rows only. The scale is not a test-fitted parameter or a numerical accuracy guarantee.",
    flow:["Training increments", "Fit species transition scales", "Signed asinh coordinate", "Coordinate z"]},
  {id:"budget-log", name:"Tolerance-scale signed-log increment", formula:"z = sign(d) * log1p(r * abs(d) / a) / r",
    detail:"Use a=1e-15 and r=0.1. The derivative is exactly 1/(a+r*abs(d)); this aligns small coordinate errors locally with the chosen increment budget.",
    flow:["Reference increment d", "Fixed tolerance scales a and r", "Signed log1p", "Coordinate z"]},
  {id:"budget-asinh", name:"Tolerance-scale asinh increment", formula:"z = asinh(r * d / a) / r",
    detail:"Use a=1e-15 and r=0.1. The derivative is 1/sqrt(a^2+r^2*d^2), not exactly the additive acceptance rule.",
    flow:["Reference increment d", "Fixed tolerance scales a and r", "Signed asinh", "Coordinate z"]},
  {id:"gbct", name:"GBCT transformed-state rate", formula:"z = sign(q) * abs(q)^0.5 / 0.5; q = [B(Y+d)-B(Y)]/h; B power = 0.1",
    detail:"Matched target adaptation of GBCT, with h=1e-6 s. Stable state difference, signed square-root rate, then train-only mean/RMS normalization. Not a reproduction of the published network, dataset or loader normalization.",
    flow:["Initial state Y and reference increment d", "Stable Box-Cox difference, exponent 0.1", "Divide by h = 1e-6 s", "Signed power, exponent 0.5", "Train-only mean/RMS scaling"]},
];
export const targetNames = Object.fromEntries(representations.map(row=>[row.id,row.name]));
targetNames.zero = "Zero-increment control";

const denseFlow = target => ["T, pressure, composition", "Training-only input scaling", `Dense network predicts ${target}`, "Undo target scaling and coordinate transform", "Physical increment prediction"];
const baseFlow = loss => ["Verified 4k transformed-state network", `4k further updates: ${loss}`, "Same transformed-state inverse", "Physical increment prediction"];
const correctionFlow = protectedFit => ["Input state", "Frozen 4k base predicts d0", "4x256 network predicts local-asinh correction", "Scale from 1e-14 + abs(d0), not reference", "Stable inverse gives corrected increment", protectedFit ? "Training adds tail and non-regression penalties" : "Training uses physical normalized error"];
const localFlow = target => ["Input state", "Training-only scaling and rank reduction", "128 nearest training neighbors", `Cubic RBF predicts ${target}`, "Inverse coordinate gives increment"];
const inputFlow = finish => ["T, pressure, composition", "Inverse T, log pressure, log-like partial pressures", "Separate 2x32 tanh network per species", "4k coordinate warmup", finish, "Inverse asinh gives increment"];

export const recipes = [
  ...["state-boxcox","gbct"].flatMap(target=>["coordinate","increment","state"].map(objective=>({
    id:`${target}-${objective}`, name:`${targetNames[target]} — paired ${objective} loss`, target, family:"Paired target and error-scale comparison",
    detail:`Fresh seeded 4x800 GELU. 2k coordinate warmup, then 2k ${objective} updates, fresh Adam for all arms. Physical objectives use a=1e-15, r=0.1; increment uses abs(d), state uses abs(Y+d). Same batches and final checkpoint. FP32 model, FP64 inverse. Both scoring policies required.`,
    flow:["Same 10k training states and seed", `Predict ${targetNames[target]}`, "2k coordinate warmup", `2k ${objective}-loss updates`, "Stable physical increment inverse", "Score BOTH increment and state allowances"],
  }))),
  ...representations.slice(0,4).flatMap(target=>[
    {id:`original-${target.id}`, name:`${target.name} — 2k updates`, target:target.id, family:"Matched dense networks",
      detail:"4x800 GELU, FP32, L1 coordinate loss, Adam, 2,000 updates. Training-only output standardization. State count and seed belong to the run.", flow:denseFlow(target.name)},
    {id:`long-${target.id}`, name:`${target.name} — 4k updates`, target:target.id, family:"Matched dense networks",
      detail:"Same 4x800 recipe; fresh 4,000-update cosine schedule from the original initialization, not continued Adam state. 10,000 selected training rows.", flow:denseFlow(target.name)},
  ]),
  ...["budget-log","budget-asinh"].flatMap(target=>[
    {id:target, name:targetNames[target], target, family:"Tolerance-derived targets",
      detail:"4x800, 2,000 updates, L1 coordinate loss. Fixed coordinate divisor 100 and zero offset; no species-wise output standardization.", flow:denseFlow(targetNames[target])},
    {id:`physical-${target}`, name:`${targetNames[target]} + physical loss`, target, family:"Tolerance-derived targets",
      detail:"Same recipe plus 0.01 times mean log1p(abs(physical error)/budget). This is a target-and-loss method, not a new representation.", flow:denseFlow(targetNames[target])},
  ]),
  {id:"residual-state-boxcox", name:"RMS-scaled additive residual", family:"Residual corrections", target:"residual",
    detail:"Freeze the original 2k base. Fit one 4x800 network for 2k updates to FP64 physical residuals, divided by per-species training RMS. Reconstruct and add in FP64. This is not the later base-relative correction.",
    flow:["Frozen 2k base predicts d0", "Training residual = reference minus d0", "Training-only species RMS scale", "4x800 network predicts normalized residual", "FP64 d0 plus predicted residual"]},
  {id:"deep-state-boxcox", name:"Transformed-state increment — 8-layer control", family:"Capacity control", target:"state-boxcox",
    detail:"8x800 GELU, 2,000 updates. Approximate parameter/work control for two networks; measured CPU time is not forced equal.", flow:denseFlow("transformed-state increment")},
  {id:"continue-coordinate", name:"Transformed-state coordinate continuation", family:"Warm-start loss changes", target:"state-boxcox",
    detail:"4,000 additional updates on the verified 4k base with its coordinate loss; 8k total updates.", flow:baseFlow("coordinate loss")},
  {id:"finetune-physical", name:"Transformed-state physical fine-tuning", family:"Warm-start loss changes", target:"state-boxcox",
    detail:"4,000 additional updates, mean log1p(abs(physical error)/budget). Output representation stays unchanged.", flow:baseFlow("physical loss")},
  {id:"finetune-tail", name:"Transformed-state tail fine-tuning", family:"Warm-start loss changes", target:"state-boxcox",
    detail:"Physical fine-tuning plus 0.25 times the mean worst-eight-species loss per state. Not the ADA-CVaR sampling algorithm.", flow:baseFlow("physical plus tail loss")},
  {id:"relative-correction", name:"Base-relative asinh correction", family:"Residual corrections", target:"base-relative-asinh",
    detail:"Frozen 4k base, zero-initialized 4x256 correction, 4k updates. Scale = 1e-14 + abs(base prediction), available at inference. It does not use reference-dependent decoding.", flow:correctionFlow(false)},
  {id:"protected-correction", name:"Protected base-relative asinh correction", family:"Residual corrections", target:"base-relative-asinh",
    detail:"Same correction plus tail loss and training-only penalty for damaging previously passing components. This does not guarantee non-regression on new states.", flow:correctionFlow(true)},
  {id:"local-state", name:"Local RBF — transformed-state increment", family:"Local tables", target:"state-boxcox",
    detail:"128-neighbor cubic radial-basis interpolation, degree-one polynomial, smoothing 1e-8. Training-only numerical-rank basis. Not ISAT or certified tabulation.", flow:localFlow("transformed-state increment")},
  {id:"local-asinh", name:"Local RBF — tolerance-scale asinh", family:"Local tables", target:"budget-asinh",
    detail:"Same local table with asinh(d/1e-14) coordinates. Fixed multiplicative coordinate factors do not change the reconstructed interpolant. Query cost includes the local solve.", flow:localFlow("tolerance-scale asinh")},
  {id:"arrhenius-heads", name:"Log-partial-pressure heads — Adam", family:"Input and architecture changes", target:"budget-asinh",
    detail:"Separate 2x32 tanh species heads. 4k coordinate updates then 4k physical updates. Standardized asinh(d/1e-14). The historical arrhenius ID denotes motivated inputs, not an exact Arrhenius law.", flow:inputFlow("4k Adam physical-loss updates")},
  {id:"arrhenius-lbfgs", name:"Log-partial-pressure heads — L-BFGS/RMS", family:"Input and architecture changes", target:"budget-asinh",
    detail:"Same coordinate warmup, then 80 full-batch L-BFGS steps, maximum 400 closure evaluations, species-RMS physical objective. Changes optimizer and loss together.", flow:inputFlow("80 full-batch L-BFGS/RMS steps")},
  {id:"arrhenius-local", name:"Log-partial-pressure local RBF", family:"Input and architecture changes", target:"budget-asinh",
    detail:"New inverse-T/log-partial-pressure inputs with the same local asinh interpolation. It has no neural training or base predictor.", flow:["Input state", "Inverse T and log-like partial pressures", "Training-only scaling and rank reduction", "128-neighbor asinh RBF", "Inverse asinh gives increment"]},
  {id:"rate-euler", name:"Initial-rate Euler control", family:"Non-learned kinetics controls", target:"physical",
    detail:"Actual mechanism evaluation at inference. One explicit fixed step, no learned weights and no adaptive error control.", flow:["Input state", "Mechanism gives initial net species rates", "Multiply rates by interval", "Physical increment prediction"]},
  {id:"frozen-exponential", name:"Frozen production–destruction control", family:"Non-learned kinetics controls", target:"physical",
    detail:"Freeze species production and destruction coefficients during the interval; use stable expm1 reconstruction. Positivity does not imply conservation or accuracy.", flow:["Input state", "Mechanism gives production and destruction", "Freeze each species coefficient", "Analytic scalar exponential step", "Physical increment prediction"]},
];
// A report-only alias for the already registered 4k base, never a new method.
export const recipeAliases = {base:"long-state-boxcox"};
export const recipeNames = Object.fromEntries(recipes.map(row=>[row.id,row.name]));
for (const [alias,id] of Object.entries(recipeAliases)) recipeNames[alias] = recipeNames[id];

export function methodLink(id) {
  const canonical = recipeAliases[id] || id;
  if (!recipeNames[canonical]) throw new Error(`Unknown method recipe: ${id}`);
  return `${catalogueUrl}#${canonical}`;
}
