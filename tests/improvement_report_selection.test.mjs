import test from "node:test";
import assert from "node:assert/strict";
import {improvementGroup, improvementSelection} from "../benchmarks/flame_conditioning/report-content/improvement-selection.mjs";

test("comparison groups retain controls independent of scores", ()=>{
  const rows = ["base","continue-coordinate","finetune-physical","finetune-tail","local-state"].map((name,index)=>({name,seed:1,componentRate:index/10}));
  assert.deepEqual(improvementSelection(rows,"objective",1).map(row=>row.name), improvementGroup("objective"));
  assert.deepEqual(improvementSelection(rows,"objective",2), []);
  assert.throws(()=>improvementGroup("unknown"));
});
