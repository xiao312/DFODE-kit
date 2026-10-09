import test from "node:test";
import assert from "node:assert/strict";
import {comparisonNames, selectComparison} from "../benchmarks/flame_conditioning/report-content/refinement-selection.mjs";

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
