import assert from "node:assert/strict";
import test from "node:test";
import { policyBinTable } from "../benchmarks/flame_conditioning/report-content/policy-bins.mjs";

test("fixed policies, population, ordering and empty bins are preserved", () => {
  const bin = {population:"balanced", lower:305, upper:500, cells:32, components:1856, p99:2};
  const input = [
    {...bin, name:"run--state-boxcox"}, {...bin, name:"run--signed-power", p99:3},
    {...bin, name:"run--fixed-hybrid", p99:4}, {...bin, name:"zero-baseline", p99:5},
    {...bin, name:"run--budget-linear", p99:0.1}, {...bin, name:"other--state-boxcox", p99:0.2},
    {...bin, name:"run--state-boxcox", population:"uniform", p99:100},
    {...bin, name:"run--state-boxcox", lower:0, upper:305, cells:0, components:0, p99:null},
  ];
  const {rows, sourceRows} = policyBinTable(input, "balanced", "run", true);
  assert.equal(sourceRows.length, 5);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].count, 0);
  assert.equal(rows[0].boxcox, null);
  assert.equal(rows[1].range, "[305, 500) K");
  assert.deepEqual([rows[1].zero, rows[1].boxcox, rows[1].power, rows[1].hybrid], [5, 2, 3, 4]);
  assert.throws(() => policyBinTable([...input, input[0]], "balanced", "run", true), /Duplicate/);
  assert.throws(() => policyBinTable([input[0], {...input[1], cells:31}], "balanced", "run", true), /populations/);
});

test("magnitude bins count species components and keep an open upper bound", () => {
  const {rows} = policyBinTable([{population:"uniform", name:"zero-baseline", lower:1e-5,
    upper:null, cells:null, components:123, p99:2e-8}], "uniform", "run", false);
  assert.equal(rows[0].range, "≥ 1e-5");
  assert.equal(rows[0].count, 123);
  assert.equal(rows[0].zero, 2e-8);
  assert.equal(rows[0].hybrid, null);
});
