export {recipeNames as improvementNames} from "./method-catalogue.mjs";
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
