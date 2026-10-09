import React, {useState} from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {recipeNames, methodLink} from "./method-catalogue.mjs";
const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;
const number = value => value.toPrecision(4);

export function FuelMatchedWork() {
  const {snapshot, visible} = useDataApp();
  const [policy, setPolicy] = useState("increment-reference-v1");
  const [seed, setSeed] = useState(20261011);
  const [recipe, setRecipe] = useState("fuel-state-matched-work");
  const models = snapshot.queries.fuel_matched_models?.rows;
  if (!models) return null;
  const allPairs = snapshot.queries.fuel_matched_pairs.rows;
  const pairs = allPairs.filter(r=>r.policy === policy);
  const selected = models.filter(r=>r.policy === policy);
  const history = snapshot.queries.fuel_matched_history.rows.filter(r=>r.policy === policy && r.seed === seed && r.recipe === recipe);
  const range = id => {
    const changes = allPairs.filter(r=>r.policy === "increment-reference-v1" && r.recipe === id).map(r=>r.changePoints);
    return `${Math.min(...changes).toFixed(2)} to ${Math.max(...changes).toFixed(2)} percentage points`;
  };
  const withNames = rows => rows.map(r=>({...r, method:recipeNames[r.recipe]}));
  const method = {field:"method",label:"Recipe",renderCell:(_,r)=><a href={methodLink(r.recipe)}>{r.method}</a>};
  const text = (id,value) => visible(id) && <ReportSection id={id} queryId="fuel_matched_models" queryIds={["fuel_matched_models","fuel_matched_pairs"]} title={id} showHeading={false} sourceRowsByQuery={{fuel_matched_models:models,fuel_matched_pairs:allPairs}}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  return <section aria-label="Matched-update dataset comparison">
    {text("matched-opening", `# Does more data help when training work stays fixed?\n\n## Executive Summary\n\nEight GPU fits are complete and verified. Each received **6,000 updates and 60 million row presentations**. The only planned difference within each recipe and seed is access to **50k or 200k unique training rows**.\n\nFor the primary increment rule, the change from 50k to 200k is **${range("fuel-state-matched-work")}** for transformed-state targets and **${range("fuel-power-matched-work")}** for direct-power targets. A positive change means 200k passed more components. Both seeds are shown below.\n\n${models.filter(r=>r.policy === "increment-reference-v1").every(r=>r.stateRate === 0) ? "No complete development state passes the primary increment rule." : "Complete-state acceptance is listed separately below."} These are development results, not independent-test or solver-level accuracy results.`)}
    {text("matched-protocol", `## Read this comparison in four steps\n\n1. **Keep the problem fixed.** Reuse the checked labels and all 1,023 development states. The 200k pool contains the 50k pool. No new labels or CFD cases are generated.\n2. **Keep training work fixed.** Use 10k rows per update for 6,000 updates. The 50k model sees 1,200 full-pool passes; the 200k model sees 300. Both process 60M row presentations.\n3. **Keep preprocessing fixed.** Fit input and target scales on the 50k training pool only. Reuse the exact arrays at 200k. Pair initial weights by seed. Learning-rate boundaries are fixed by update, not epoch.\n4. **Evaluate the final model.** Do not choose a checkpoint from development scores. Check saved-model replay, independent pass counts, physical diagnostics, and all paired identities.\n\nThe recipes still use different target normalization and Adam schedules. This isolates the data-size effect **within a recipe**, not the effect of coordinates alone. It is a controlled Fuel-derived experiment, not a literal repeat of the source epoch schedules. Fit wall time includes full-training-pool diagnostics, which cost more at 200k.`)}
    <div style={{display:"flex",flexWrap:"wrap",gap:"1rem",marginBlock:"1rem"}} aria-label="Matched-work controls">
      <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{["increment-reference-v1","state-endpoint-v1"].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve seed <select value={seed} onChange={e=>setSeed(Number(e.target.value))}>{[20261011,20261012].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve recipe <select value={recipe} onChange={e=>setRecipe(e.target.value)}>{["fuel-state-matched-work","fuel-power-matched-work"].map(v=><option key={v} value={v}>{recipeNames[v]}</option>)}</select></label>
    </div>
    {visible("matched-pairs") && <DataComponent id="matched-pairs" queryId="fuel_matched_pairs" kind="table" title={`200k minus 50k — ${policy}, a=1e-15, r=0.1`} sourceRows={pairs} displayRows={withNames(pairs)}><DataTable rows={withNames(pairs)} searchable={false} label="Every paired data-size change" columns={[method,{field:"seed",label:"Seed"},{field:"smallRate",label:"50k development pass",renderCell:percent},{field:"largeRate",label:"200k development pass",renderCell:percent},{field:"changePoints",label:"Change, percentage points",renderCell:v=>`${v >= 0 ? "+" : ""}${v.toFixed(2)}`} ]}/></DataComponent>}
    {visible("matched-history") && <EvidenceChart id="matched-history" queryId="fuel_matched_history" title={`Equal-update learning curves — ${recipeNames[recipe]}, seed ${seed}`} rows={history} sourceRows={history} height={360} spec={{type:"line",x:"updates",y:"acceptanceRate",series:"series",stackable:false,xLabel:"Completed weight updates",yLabel:"Component acceptance",valueDecimals:2}}/>}
    {visible("matched-models") && <DataComponent id="matched-models" queryId="fuel_matched_models" kind="table" title={`All final matched-work models — ${policy}`} sourceRows={selected} displayRows={withNames(selected)}><DataTable rows={withNames(selected)} label="All matched-work fits, no best-seed selection" columns={[
      method,{field:"seed",label:"Seed"},{field:"trainingCount",label:"Unique training rows"},
      {field:"trainingRate",label:"Training pass",renderCell:percent},{field:"developmentRate",label:"Development pass",renderCell:percent},
      {field:"stateRate",label:"Complete development states",renderCell:percent},{field:"p99",label:"Error / allowance p99",renderCell:number},
      {field:"updates",label:"Updates"},{field:"presentations",label:"Row presentations"},{field:"fitSeconds",label:"GPU fit wall s",renderCell:number},
      {field:"negativeRate",label:"Negative endpoints",renderCell:percent},{field:"correctionRate",label:"Inverse corrections",renderCell:percent},
      {field:"independentTest",label:"Independent test",renderCell:percent},
    ]}/></DataComponent>}
    {text("matched-next", `## More unique rows help, but they do not resolve the accuracy problem\n\nBoth recipes improve on development states at equal work, in both seeds. The gain is only about 2–3 percentage points for four times as many unique rows. At 200k, training and development scores are close. Low accuracy is therefore not explained by a large train-to-development gap alone. This does not prove that either recipe has reached its best attainable accuracy.\n\nTransformed-state targets pass more components, but their normalized error tails remain much larger than those of direct power. Direct power still produces negative endpoint components. There is no single winner on all accuracy and physical checks.\n\nThe earlier fixed-epoch results below use more updates and refit their scales at each size. These controlled fits do **not** replace those higher-budget models or indicate a regression. ${snapshot.queries.fuel_budget_models ? "The completed fixed-data budget comparison above tests the next question: whether more training helps on the same 200k pool." : "Before 1M rows, test a larger update budget on the existing 200k pool while keeping the new control's scalers, batches, development states and pass rules fixed."}\n\nThe acceptance rule has not changed: **abs(error) ≤ 1e-15 + 0.1 × abs(reference increment)**. The state policy changes the scale to the reference endpoint; it answers a different accuracy question.`)}
  </section>;
}
