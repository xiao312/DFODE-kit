import test from "node:test";
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {recipes, representations, recipeNames, recipeAliases, methodLink} from "../benchmarks/flame_conditioning/report-content/method-catalogue.mjs";
import {improvementGroup, improvementNames} from "../benchmarks/flame_conditioning/report-content/improvement-selection.mjs";
import {comparisonNames, refinementNames} from "../benchmarks/flame_conditioning/report-content/refinement-selection.mjs";
import {renderCatalogue} from "../scripts/build_research_method_catalogue.mjs";
import {annotateReview} from "../scripts/prepare_research_review.mjs";

test("catalogue IDs and names are unambiguous within each type",()=>{
  for (const rows of [recipes,representations]) {
    assert.equal(new Set(rows.map(row=>row.id)).size,rows.length);
    assert.equal(new Set(rows.map(row=>row.name)).size,rows.length);
    for (const row of rows) assert.ok(row.flow.length >= 3 && row.detail.length > 20);
  }
  assert.equal(recipeAliases.base,"long-state-boxcox");
  assert.throws(()=>methodLink("unknown"),/Unknown/);
});

test("comparison helpers share the registry and all displayed methods resolve",()=>{
  assert.equal(improvementNames,recipeNames);
  assert.equal(refinementNames,recipeNames);
  const methods = ["objective","correction","local","inputs"].flatMap(improvementGroup);
  for (const target of representations.slice(0,4)) methods.push(...comparisonNames("training",target.id));
  for (const group of ["coordinates","loss","residual"]) methods.push(...comparisonNames(group));
  methods.push("rate-euler","frozen-exponential");
  for (const id of methods) {
    assert.ok(recipeNames[id],id);
    assert.match(methodLink(id), /research-method-catalogue\.md#/);
  }
});

test("generated GitHub diagrams and alias catalogue are current",()=>{
  const actual = readFileSync(new URL("../docs/research-method-catalogue.md",import.meta.url),"utf8").replaceAll("\r\n","\n");
  assert.equal(actual,renderCatalogue());
  for (const row of recipes) assert.ok(actual.includes(`<a id="${row.id}"></a>`));
  assert.equal((actual.match(/```mermaid/g) || []).length,1+recipes.length+representations.length);
});

test("split metadata preserves all evidence, unknowns and artifact identity",()=>{
  const query = {rows:[{name:"long-state-boxcox",componentRate:0,trainingComponentRate:null}],source:{metricDefinitions:[]}};
  const source = {id:"stable",generatedAt:"original",queries:{improve_models:query,refinement_models:query}};
  const result = annotateReview(source);
  assert.equal(result.id,source.id);
  assert.equal(result.generatedAt,source.generatedAt);
  assert.deepEqual(result.queries.improve_models.rows,source.queries.improve_models.rows);
  assert.equal(source.queries.improve_models.source.metricDefinitions.length,0);
  assert.deepEqual(annotateReview(result),result);
  assert.throws(()=>annotateReview({queries:{}}),/Missing/);
});
