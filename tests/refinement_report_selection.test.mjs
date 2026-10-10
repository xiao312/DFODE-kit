import test from "node:test";
import assert from "node:assert/strict";
import {comparisonNames, residualBinComparison, selectComparison} from "../benchmarks/flame_conditioning/report-content/refinement-selection.mjs";

test("comparison membership does not depend on accuracy", () => {
  for (const group of ["training","coordinates","loss","residual"]) {
    const names = comparisonNames(group,"state-boxcox");
    const rows = [1,2].flatMap(seed=>names.map((name,index)=>({seed,name,componentRate:index/10})));
    assert.equal(selectComparison(rows,names,1).length,names.length);
    assert.deepEqual(selectComparison(rows,names,1).map(row=>row.name), names);
    assert.throws(()=>selectComparison(rows.slice(1),names,1), /Missing/);
    assert.throws(()=>selectComparison([...rows,rows[0]],names,1), /duplicate/);
  }
});

test("residual-bin comparison preserves fixed populations and nulls", () => {
  const base = [{seed:1,trainingCount:10000,target:"state-boxcox",lower:0,upper:1e-15,components:8,pass_fraction:.75},
    {seed:1,trainingCount:10000,target:"state-boxcox",lower:1e-15,upper:null,components:0,pass_fraction:null}];
  const residual = base.map(row=>({...row,name:"residual-state-boxcox",pass_fraction:row.components ? .5 : null}));
  const rows = residualBinComparison(base,residual,1);
  assert.equal(rows[0].baseRate,.75);
  assert.equal(rows[0].residualRate,.5);
  assert.equal(rows[1].residualRate,null);
  assert.throws(()=>residualBinComparison(base,[...residual,residual[0]],1),/duplicated/);
  assert.throws(()=>residualBinComparison(base,residual.map(row=>({...row,components:99})),1),/populations/);
});
