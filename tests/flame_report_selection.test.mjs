import assert from "node:assert/strict";
import test from "node:test";
import { largestPrimaryComparison } from "../benchmarks/flame_conditioning/report-content/validation-selection.mjs";

const targets = ["state-boxcox", "signed-power", "budget-linear", "scaled-asinh"];
const comparison = (trainingCount, score) => targets.map(target => ({target, trainingCount,
  seed:20261009, precision:"float32", architecture:"800x800x800x800", activation:"gelu",
  loss:"l1", updates:10000, budgetP99:score}));

test("choose largest matched size even when all its scores are worse", () => {
  const small = comparison(49979, 1), large = comparison(199000, 100);
  const otherSeed = {...large[0], seed:20261010, trainingCount:300000, budgetP99:0};
  const input = [...large, ...small, otherSeed].reverse();
  const copy = JSON.stringify(input);
  assert.deepEqual(largestPrimaryComparison(input), large);
  assert.equal(JSON.stringify(input), copy);
});

test("do not silently fall back from an incomplete or duplicate largest comparison", () => {
  const small = comparison(49979, 1), large = comparison(199000, 100);
  assert.throws(() => largestPrimaryComparison([...small, ...large.slice(1)]), /four targets/);
  assert.throws(() => largestPrimaryComparison([...large, large[0]]), /four targets/);
  assert.throws(() => largestPrimaryComparison([{...large[0], trainingCount:NaN}]), /Invalid training count/);
});

test("empty input and other recipes do not create a comparison", () => {
  assert.deepEqual(largestPrimaryComparison([]), []);
  for (const override of [{precision:"float64"}, {updates:2000}, {activation:"tanh"},
    {loss:"mse"}, {architecture:"128x128x128"}]) {
    assert.deepEqual(largestPrimaryComparison(comparison(49979, 1).map(row => ({...row, ...override}))), []);
  }
});
