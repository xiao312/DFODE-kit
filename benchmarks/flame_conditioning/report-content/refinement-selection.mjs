// Comparison membership is fixed by method, never by observed accuracy.
export const refinementNames = {
  "original-state-boxcox": "Transformed state · 2k updates",
  "original-signed-power": "Signed power · 2k updates",
  "original-budget-linear": "State-budget linear · 2k updates",
  "original-scaled-asinh": "Empirical-scale asinh · 2k updates",
  "long-state-boxcox": "Transformed state · 4k updates",
  "long-signed-power": "Signed power · 4k updates",
  "long-budget-linear": "State-budget linear · 4k updates",
  "long-scaled-asinh": "Empirical-scale asinh · 4k updates",
  "budget-log": "Budget signed-log",
  "budget-asinh": "Budget asinh",
  "physical-budget-log": "Budget signed-log + physical loss",
  "physical-budget-asinh": "Budget asinh + physical loss",
  "residual-state-boxcox": "Frozen base + residual",
  "deep-state-boxcox": "Larger single model · 8 layers",
};

export function comparisonNames(group, target) {
  if (group === "training") return [`original-${target}`, `long-${target}`];
  if (group === "coordinates") return ["original-state-boxcox", "original-scaled-asinh", "budget-log", "budget-asinh"];
  if (group === "loss") return ["budget-log", "physical-budget-log", "budget-asinh", "physical-budget-asinh"];
  if (group === "residual") return ["original-state-boxcox", "long-state-boxcox", "residual-state-boxcox", "deep-state-boxcox"];
  throw new Error("Unknown refinement comparison");
}

export function selectComparison(rows, names, seed) {
  const selected = rows.filter(row => row.seed === seed && names.includes(row.name));
  if (selected.length !== names.length || new Set(selected.map(row => row.name)).size !== names.length) {
    throw new Error("Missing or duplicate comparison model");
  }
  return selected.map(row => ({...row, method:refinementNames[row.name]}));
}
