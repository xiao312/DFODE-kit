import React, { useState } from "react";
import { DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
import { comparisonNames, refinementNames, selectComparison } from "./refinement-selection.mjs";

const percent = value => value == null ? "Unknown" : `${(100*value).toFixed(2)}%`;
const numeric = value => value == null ? "Unknown" : value.toLocaleString("en-US", {maximumFractionDigits:2});

export function Refinement() {
  const {snapshot, visible} = useDataApp();
  const [seed, setSeed] = useState(20261011);
  const [group, setGroup] = useState("coordinates");
  const [target, setTarget] = useState("state-boxcox");
  const [floor, setFloor] = useState(1e-15);
  const [measure, setMeasure] = useState("componentRate");
  const [cost, setCost] = useState("trainingSeconds");
  const models = snapshot.queries.refinement_models?.rows;
  if (!models) return null;
  const original = snapshot.queries.offline_models.rows.filter(row=>row.trainingCount === 10000).map(row=>({
    ...row, name:`original-${row.target}`, inferenceMs:1000*row.inferenceSeconds/row.states,
    trainingComponentRate:null, trainingStateRate:null, parameters:null, negativeRate:row.negativeComponentRate,
  }));
  const all = [...original, ...models];
  const selectedNames = comparisonNames(group, target);
  const selected = selectComparison(all, selectedNames, seed);
  const modelSources = {
    refinement_models:models.filter(row=>row.seed === seed && selectedNames.includes(row.name)),
    offline_models:snapshot.queries.offline_models.rows.filter(row=>row.seed === seed && row.trainingCount === 10000 && selectedNames.includes(`original-${row.target}`)),
  };
  const priorCurves = snapshot.queries.offline_curves.rows.filter(row=>row.trainingCount === 10000).map(row=>({...row,name:`original-${row.target}`}));
  const curves = [...priorCurves,...snapshot.queries.refinement_curves.rows].filter(row=>
    row.seed === seed && row.atol === floor && selectedNames.includes(row.name));
  const curveRows = curves.map(row=>({...row,method:refinementNames[row.name]})).sort((a,b)=>a.rtol-b.rtol);
  const curveSources = {
    refinement_curves:snapshot.queries.refinement_curves.rows.filter(row=>row.seed === seed && row.atol === floor && selectedNames.includes(row.name)),
    offline_curves:snapshot.queries.offline_curves.rows.filter(row=>row.seed === seed && row.trainingCount === 10000 && row.atol === floor && selectedNames.includes(`original-${row.target}`)),
  };
  const histories = snapshot.queries.refinement_history.rows.filter(row=>row.seed === seed && selectedNames.includes(row.name));
  const bins = snapshot.queries.refinement_bins.rows.filter(row=>row.seed === seed && selectedNames.includes(row.name));
  const audits = snapshot.queries.refinement_audit.rows.filter(row=>row.seed === seed && selectedNames.includes(row.name));
  const species = snapshot.queries.refinement_species.rows.filter(row=>row.seed === seed && selectedNames.includes(row.name));
  const binRows = bins.map(row=>({...row,method:refinementNames[row.name],range:`[${row.lower.toExponential(0)}, ${row.upper == null ? "∞" : row.upper.toExponential(0)})`}));
  const auditRows = audits.map(row=>({...row,method:refinementNames[row.name]}));
  const speciesRows = species.map(row=>({...row,method:refinementNames[row.name]}));
  const text = (id, query, value) => visible(id) && <ReportSection id={id} queryId={query} title={id} showHeading={false} sourceRows={snapshot.queries[query].rows}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  const maxState = Math.max(...models.map(row=>row.stateRate));
  const bestComponent = Math.max(...models.map(row=>row.componentRate));
  const range = name => {
    const values = all.filter(row=>row.name === name).map(row=>row.componentRate);
    return `${percent(Math.min(...values))}–${percent(Math.max(...values))}`;
  };
  const outcomes = [
    `Longer conventional training: ${range("original-state-boxcox")} → ${range("long-state-boxcox")}.`,
    `Tolerance-derived targets: signed-log ${range("budget-log")}; asinh ${range("budget-asinh")}.`,
    `Added physical loss: signed-log ${range("physical-budget-log")}; asinh ${range("physical-budget-asinh")}.`,
    `Frozen-base residual: ${range("residual-state-boxcox")}; larger single-model control: ${range("deep-state-boxcox")}.`,
  ];
  return <section aria-label="Representation refinement">
    {text("refinement-opening", "refinement_models", `## Current stage: test how to recover more accurate predictions\n\nTwenty new fits completed: ten fixed methods for each of two seeds. The highest observed species-component acceptance is ${percent(bestComponent)}. The highest complete-state acceptance is ${percent(maxState)}. These are descriptive maxima, not a selected deployment model. The required whole-state research milestone remains 99%.\n\nThe dataset and the pass rule did not change. Every prediction is judged against |error| ≤ 10⁻¹⁵ + 0.1 × |reference increment|. We have not started a new CFD test.`)}
    {visible("refinement-summary") && <ReportSection id="refinement-summary" queryId="refinement_models" queryIds={["refinement_models","offline_models"]} sourceRowsByQuery={{refinement_models:models,offline_models:snapshot.queries.offline_models.rows.filter(row=>row.trainingCount === 10000)}} showHeading={false} title="Executive Summary">
      <RichNarrative id="refinement-summary:body" value={`### Executive Summary\n\nThese are component pass rates at the unchanged primary tolerance. Each range retains both seeds; it is not a confidence interval.\n\n${outcomes.map(value=>`- ${value}`).join("\n")}\n\nNo component result alone establishes a usable full-state chemistry solver. The plots below show the tolerance and cost trade-offs.`}/>
    </ReportSection>}
    {text("refinement-method", "refinement_models", `### What each change tests\n\n1. **More training:** restart each original formulation for 4,000 updates instead of 2,000. The learning-rate schedule also lasts longer. This checks a larger optimization budget; it does not prove that training has saturated.\n2. **Tolerance-derived targets:** replace the empirical transition scale with 10⁻¹⁴, derived from the absolute and relative budgets. Signed-log and asinh give small and large changes different weights. They use one fixed output scale, not separate standardization for each species. This tests target-plus-loss conditioning.\n3. **Physical loss:** add a robust penalty on error divided by the same physical budget. Keep each new target, model and update count unchanged. This tests whether the training objective helps.\n4. **Residual correction:** freeze the original transformed-state model. Train one new network to predict its remaining error. Add the two predictions in FP64. Compare with longer training and a larger single network. FP64 addition does not make an FP32 network a double-precision predictor.\n\nAll fits use the same 10k training rows within each seed. All scales use training rows only. Both seeds are retained. The evaluation snapshots are development data from one flame, not an untouched test.`)}
    <div style={{display:"flex",gap:"1rem",flexWrap:"wrap",marginBlock:"1rem"}} aria-label="Refinement comparison controls">
      <label>Question <select value={group} onChange={event=>setGroup(event.target.value)}><option value="coordinates">Tolerance-derived targets</option><option value="loss">Added physical loss</option><option value="residual">Residual versus single model</option><option value="training">Longer training</option></select></label>
      {group === "training" && <label>Original target <select value={target} onChange={event=>setTarget(event.target.value)}>{["state-boxcox","signed-power","budget-linear","scaled-asinh"].map(value=><option key={value} value={value}>{value}</option>)}</select></label>}
      <label>Seed <select value={seed} onChange={event=>setSeed(Number(event.target.value))}>{[20261011,20261012].map(value=><option key={value} value={value}>{value}</option>)}</select></label>
      <label>Absolute floor <select value={floor} onChange={event=>setFloor(Number(event.target.value))}>{[1e-12,1e-15,1e-18].map(value=><option key={value} value={value}>{value.toExponential(0)}</option>)}</select></label>
      <label>Pass criterion <select value={measure} onChange={event=>setMeasure(event.target.value)}><option value="componentRate">Species components</option><option value="stateRate">Complete states</option></select></label>
    </div>
    {visible("refinement-tolerance") && <EvidenceChart id="refinement-tolerance" queryId="refinement_curves" queryIds={["refinement_curves","offline_curves"]} title={`Acceptance versus tolerance — seed ${seed}, floor ${floor.toExponential(0)}`} rows={curveRows} displayRows={curveRows} sourceRowsByQuery={curveSources} height={340}
      spec={{type:"line",x:"relativeLabel",y:measure,series:"method",stackable:false,valueDecimals:2,yLabel:"Acceptance",xLabel:"Allowed relative error"}}/>}
    <label>Cost <select value={cost} onChange={event=>setCost(event.target.value)}><option value="trainingSeconds">Training CPU seconds</option><option value="inferenceMs">Inference CPU ms per state</option></select></label>
    {visible("refinement-cost") && <EvidenceChart id="refinement-cost" queryId="refinement_models" queryIds={["refinement_models","offline_models"]} title="Acceptance versus measured cost — primary tolerance" rows={selected} displayRows={selected} sourceRowsByQuery={modelSources} height={340}
      spec={{type:"scatter",x:cost,y:measure,series:"method",stackable:false,valueDecimals:3,xLabel:cost === "trainingSeconds" ? "Training CPU seconds" : "Inference CPU ms per state",yLabel:"Acceptance"}}/>}
    {visible("refinement-table") && <DataComponent id="refinement-table" queryId="refinement_models" kind="table" title="All new fits — primary tolerance; both seeds retained" sourceRows={models} displayRows={models.map(row=>({...row,method:refinementNames[row.name]}))}>
      <DataTable rows={models.map(row=>({...row,method:refinementNames[row.name]}))} label="All refinement fits" compactNumbers={false} columns={[
        {field:"method",label:"Method"},{field:"seed",label:"Seed"},{field:"trainingComponentRate",label:"Train components",renderCell:percent},
        {field:"componentRate",label:"Evaluation components",renderCell:percent},{field:"stateRate",label:"Evaluation states",renderCell:percent},
        {field:"sspiRate",label:"SSPI",renderCell:percent},{field:"trainingSeconds",label:"CPU s incl. base",renderCell:numeric},
        {field:"inferenceMs",label:"Inference CPU ms/state",renderCell:value=>value.toPrecision(3)},
        {field:"parameters",label:"Total parameters",renderCell:numeric},{field:"negativeRate",label:"Negative endpoints",renderCell:percent},
      ]}/>
    </DataComponent>}
    {visible("refinement-history") && <EvidenceChart id="refinement-history" queryId="refinement_history" title="New-model learning curves — primary tolerance, fixed final checkpoint" rows={histories.map(row=>({...row,method:refinementNames[row.name]}))} sourceRows={histories} height={320}
      spec={{type:"line",x:"step",y:measure,series:"method",stackable:false,valueDecimals:2,xLabel:"Optimizer updates",yLabel:"Evaluation acceptance"}}/>}
    {visible("refinement-bins") && <DataComponent id="refinement-bins" queryId="refinement_bins" kind="table" title="New-model acceptance by increment magnitude — primary tolerance" sourceRows={bins} displayRows={binRows}>
      <DataTable rows={binRows} label="Magnitude bins" columns={[
        {field:"method",label:"Method"},{field:"range",label:"|Reference increment|"},{field:"components",label:"Components",renderCell:numeric},{field:"pass_fraction",label:"Pass",renderCell:percent},
      ]}/>
    </DataComponent>}
    {visible("refinement-audit") && <DataComponent id="refinement-audit" queryId="refinement_audit" kind="table" title="Independent-reference subset — 16 evaluation states, primary tolerance" sourceRows={audits} displayRows={auditRows}>
      <DataTable rows={auditRows} label="Audited subset" columns={[
        {field:"method",label:"Method"},{field:"qualified_components",label:"Qualified components"},{field:"unknown_components",label:"Unknown"},
        {field:"qualified_pass_fraction",label:"Qualified components pass",renderCell:percent},{field:"qualified_state_pass_fraction",label:"Qualified states pass",renderCell:percent},
      ]}/>
    </DataComponent>}
    {visible("refinement-species") && <DataComponent id="refinement-species" queryId="refinement_species" kind="table" title="New-model acceptance for each species — primary tolerance" sourceRows={species} displayRows={speciesRows}>
      <DataTable rows={speciesRows} label="Species acceptance" columns={[
        {field:"method",label:"Method"},{field:"species",label:"Species"},{field:"components",label:"Components"},
        {field:"component_pass_fraction",label:"Pass",renderCell:percent},
        {field:"normalized_error_p99",label:"Error / budget p99",renderCell:value=>value.toExponential(2)},
      ]}/>
    </DataComponent>}
    {text("refinement-next", "refinement_models", `### How to read this evidence\n\nA species score and a complete-state score answer different questions. Every one of the 58 non-argon species must pass for a state to pass. The zero-change baseline and the original four-target data-size study remain below. SSPI measures small-change preservation, not general accuracy.\n\nThe residual cost includes its original base model. The eight-layer control and two-network model have approximately similar capacity, not identical measured cost. Training CPU time includes each protocol's preparation and periodic checks. Inference is measured on full batches; it is not single-state latency or a speedup over CVODE.\n\nOnly 16 evaluation states have an independent reference check. Full-population scores remain nominal. Inspect the magnitude bins and negative endpoints before selecting a method. A failed method is useful evidence against this particular recipe, not proof that the whole method family cannot work.\n\nThe earlier sections below are retained as experiment records. Their recommendations describe the stage at which they were written.`)}
  </section>;
}
