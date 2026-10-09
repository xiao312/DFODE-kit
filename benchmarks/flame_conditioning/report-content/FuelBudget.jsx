import React, {useState} from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {recipeNames, methodLink} from "./method-catalogue.mjs";
const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;
const number = value => value.toPrecision(4);

export function FuelBudget() {
  const {snapshot, visible} = useDataApp();
  const [policy, setPolicy] = useState("increment-reference-v1");
  const [seed, setSeed] = useState(20261011);
  const [recipe, setRecipe] = useState("fuel-state-budget");
  const [axis, setAxis] = useState("updates");
  const models = snapshot.queries.fuel_budget_models?.rows;
  if (!models) return null;
  const allPairs = snapshot.queries.fuel_budget_pairs.rows;
  const pairs = allPairs.filter(r=>r.policy === policy);
  const selected = models.filter(r=>r.policy === policy);
  const history = snapshot.queries.fuel_budget_history.rows.filter(r=>r.policy === policy && r.seed === seed && r.recipe === recipe);
  const range = (id,field="changePoints") => {
    const values = allPairs.filter(r=>r.recipe === id && r.policy === "increment-reference-v1").map(r=>r[field]);
    return `${Math.min(...values).toFixed(2)}–${Math.max(...values).toFixed(2)}`;
  };
  const named = rows => rows.map(r=>({...r,method:recipeNames[r.recipe]}));
  const method = {field:"method",label:"Recipe",renderCell:(_,r)=><a href={methodLink(r.recipe)}>{r.method}</a>};
  const text = (id,value) => visible(id) && <ReportSection id={id} queryId="fuel_budget_models" queryIds={["fuel_budget_models","fuel_budget_pairs"]} sourceRowsByQuery={{fuel_budget_models:models,fuel_budget_pairs:allPairs}} title={id} showHeading={false}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  return <section aria-label="Fixed-data training budget">
    {text("budget-opening", `# What do three times as many training updates buy?\n\n## Executive Summary\n\nAt fixed **200k training rows**, changing from 6k to 18k updates changes development acceptance by **${range("fuel-state-budget")} percentage points** for transformed-state targets and **${range("fuel-power-budget")} points** for direct-power targets. The table retains both seeds.\n\nThe longer fits use **180M row presentations**, versus 60M. Their measured fit-time ratios are ${range("fuel-state-budget","wallRatio")}× and ${range("fuel-power-budget","wallRatio")}×, respectively. These costs include diagnostics, not reference generation or final verification.\n\n${models.filter(r=>r.policy === "increment-reference-v1" && r.updates === 18000).every(r=>r.stateRate === 0) ? "No complete development state passes the primary increment rule." : "Complete-state acceptance is listed separately below."} The independent test remains unopened.`)}
    {text("budget-protocol", `## How to read this experiment\n\n1. Keep the same 200k training rows and all 1,023 development states. Reuse the exact input and target scalers fitted on 50k training rows.\n2. Start from the same initial weights for each seed. Keep the architecture, batch size of 10k, target, loss, precision and first 6k batch selections.\n3. Increase the planned budget from **6k to 18k updates**. Stretch each learning-rate stage threefold. Keep each recipe's Adam reset policy. This tests a larger budget with a proportionally longer schedule; it is not low-rate continuation or an isolated schedule comparison.\n4. Score the final checkpoint. Check saved-model replay, independent counts, physical metrics and paired data identities. Do not choose the best development checkpoint.\n\nThe primary allowance remains **1e-15 + 0.1 × abs(reference increment)**. The state policy below instead scales by the reference endpoint. Both use the same physical increment error. Changing the selected policy is not a learning gain.`)}
    <div style={{display:"flex",flexWrap:"wrap",gap:"1rem",marginBlock:"1rem"}} aria-label="Budget comparison controls">
      <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{["increment-reference-v1","state-endpoint-v1"].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve seed <select value={seed} onChange={e=>setSeed(Number(e.target.value))}>{[20261011,20261012].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve recipe <select value={recipe} onChange={e=>setRecipe(e.target.value)}>{["fuel-state-budget","fuel-power-budget"].map(v=><option key={v} value={v}>{recipeNames[v]}</option>)}</select></label>
      <label>Curve x-axis <select value={axis} onChange={e=>setAxis(e.target.value)}><option value="updates">Weight updates</option><option value="seconds">Fit wall seconds</option></select></label>
    </div>
    {visible("budget-pairs") && <DataComponent id="budget-pairs" queryId="fuel_budget_pairs" kind="table" title={`6k versus 18k updates — ${policy}`} sourceRows={pairs} displayRows={named(pairs)}><DataTable rows={named(pairs)} searchable={false} label="Every paired training-budget change" columns={[method,{field:"seed",label:"Seed"},{field:"smallRate",label:"6k development pass",renderCell:percent},{field:"largeRate",label:"18k development pass",renderCell:percent},{field:"changePoints",label:"Change, percentage points",renderCell:v=>`${v>=0?"+":""}${v.toFixed(2)}`},{field:"wallRatio",label:"Fit wall ratio",renderCell:v=>`${v.toFixed(2)}×`} ]}/></DataComponent>}
    {visible("budget-history") && <EvidenceChart id="budget-history" queryId="fuel_budget_history" title={`Training and development — ${recipeNames[recipe]}, seed ${seed}`} rows={history} sourceRows={history} height={360} spec={{type:"line",x:axis,y:"acceptanceRate",series:"series",stackable:false,xLabel:axis === "updates" ? "Completed weight updates" : "Fit wall seconds, including diagnostics",yLabel:"Component acceptance",valueDecimals:2}}/>}
    {visible("budget-models") && <DataComponent id="budget-models" queryId="fuel_budget_models" kind="table" title={`Final models and zero control — ${policy}`} sourceRows={selected} displayRows={named(selected)}><DataTable rows={named(selected)} label="All budget fits, both seeds" columns={[
      method,{field:"seed",label:"Seed"},{field:"updates",label:"Updates"},
      {field:"trainingRate",label:"Training pass",renderCell:percent},{field:"developmentRate",label:"Development pass",renderCell:percent},
      {field:"zeroRate",label:"Zero-increment control",renderCell:percent},{field:"stateRate",label:"Complete development states",renderCell:percent},
      {field:"p99",label:"Error / allowance p99",renderCell:number},{field:"zeroP99",label:"Zero-control p99",renderCell:number},{field:"fitSeconds",label:"GPU fit wall s",renderCell:number},
      {field:"inferenceMicrosecondsPerState",label:"Batched inference µs/state",renderCell:number},
      {field:"negativeRate",label:"Negative endpoints",renderCell:percent},{field:"correctionRate",label:"Inverse corrections",renderCell:percent},
      {field:"independentTest",label:"Independent test",renderCell:percent},
    ]}/></DataComponent>}
    {text("budget-next", `## More training helps, but the error tails remain a major problem\n\nBoth recipes improve in both seeds without new rows. The 6k-update result was not a performance ceiling. The training protocol is therefore an important part of this accuracy problem. These two budget levels do not establish saturation or unlimited scaling. Between-recipe differences still mix representation, normalization and optimizer schedule.\n\nAt 18k updates, transformed-state targets pass more components, but their primary-rule error p99 remains about **33k–35k allowance units**. Direct power has a much smaller p99 of about **25–27**, but still gives **3.35–3.49% negative endpoint components**. The zero-increment control passes only **5.08%** of components under the primary rule, while its p99 is about 10. More passing components can coexist with much worse outliers. None of these observations establishes solver-level accuracy.\n\nUse the stronger fits as controls for the next matched representation and loss comparison on the same 200k pool. Retain tail and physical checks alongside acceptance; do not expand to 1M just to hide a training-protocol difference. Batched inference cost includes preprocessing, transfers and stable reconstruction, not only the network.\n\nAll earlier size comparisons remain below. No 1M dataset or CFD deployment is included in this campaign.`)}
  </section>;
}
