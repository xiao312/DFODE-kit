// Present fixed BC/PT/hybrid policies without selecting by measured performance.
export function policyBinTable(allRows, population, runName, temperature) {
  const selected = allRows.filter(row => row.population === population &&
    (row.name === "zero-baseline" || (runName && row.name.startsWith(runName + "--") &&
      ["state-boxcox", "signed-power", "fixed-hybrid"].some(suffix => row.name.endsWith(suffix)))));
  const groups = new Map();
  for (const row of selected) {
    const key = JSON.stringify([row.lower, row.upper]);
    let group = groups.get(key);
    if (!group) {
      const bound = value => value === 0 ? "0" : value.toExponential(0);
      group = {lower: row.lower, upper: row.upper,
        range: temperature ? `[${row.lower}, ${row.upper}) K` :
          row.upper == null ? `≥ ${bound(row.lower)}` : `[${bound(row.lower)}, ${bound(row.upper)})`,
        count: temperature ? row.cells : row.components,
        zero: null, boxcox: null, power: null, hybrid: null};
      groups.set(key, group);
    }
    if (group.count !== (temperature ? row.cells : row.components)) {
      throw new Error("Policy-bin populations do not match");
    }
    const field = row.name === "zero-baseline" ? "zero" :
      row.name.endsWith("state-boxcox") ? "boxcox" :
      row.name.endsWith("signed-power") ? "power" : "hybrid";
    if (Object.hasOwn(group, field + "Seen")) throw new Error("Duplicate policy-bin evidence");
    group[field] = row.p99;
    group[field + "Seen"] = true;
  }
  return {sourceRows: selected, rows: [...groups.values()].sort((a, b) => a.lower - b.lower)};
}
