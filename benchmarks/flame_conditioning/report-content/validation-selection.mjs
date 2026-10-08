// Input is the reviewed, completed-model query; scores never select the display.
export function largestPrimaryComparison(models) {
  const targets = ["state-boxcox", "signed-power", "budget-linear", "scaled-asinh"];
  const candidates = models.filter(row => row.seed === 20261009 &&
    row.precision === "float32" && row.architecture === "800x800x800x800" &&
    row.activation === "gelu" && row.loss === "l1" && row.updates === 10000);
  if (!candidates.length) return [];
  if (candidates.some(row => !Number.isInteger(row.trainingCount) || row.trainingCount <= 0)) {
    throw new Error("Invalid training count in the reviewed comparison");
  }
  const count = Math.max(...candidates.map(row => row.trainingCount));
  const rows = candidates.filter(row => row.trainingCount === count);
  if (rows.length !== targets.length || targets.some(target => rows.filter(row => row.target === target).length !== 1)) {
    throw new Error("Largest comparison must contain each of the four targets exactly once");
  }
  return targets.map(target => rows.find(row => row.target === target));
}
