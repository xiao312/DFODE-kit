export const improvementNames = {
  "base": "Conventional base — 4k updates",
  "continue-coordinate": "Continue coordinate loss",
  "finetune-physical": "Physical loss fine-tune",
  "finetune-tail": "Physical + tail fine-tune",
  "relative-correction": "Relative-scale correction",
  "protected-correction": "Protected relative correction",
  "local-state": "Local RBF — transformed state",
  "local-asinh": "Local RBF — tolerance asinh",
  "arrhenius-heads": "Arrhenius inputs — species heads",
  "arrhenius-lbfgs": "Arrhenius heads — L-BFGS/RMS",
  "arrhenius-local": "Arrhenius inputs — local RBF",
};
export function improvementGroup(group) {
  const groups = {
    objective:["base", "continue-coordinate", "finetune-physical", "finetune-tail"],
    correction:["base", "relative-correction", "protected-correction"],
    local:["base", "local-state", "local-asinh"],
    inputs:["base", "arrhenius-heads", "arrhenius-lbfgs", "arrhenius-local"],
  };
  if (!groups[group]) throw new Error("Unknown comparison group");
  return groups[group];
}
export function improvementSelection(rows, group, seed) {
  const names = improvementGroup(group);
  return rows.filter(row=>row.seed === seed && names.includes(row.name));
}
