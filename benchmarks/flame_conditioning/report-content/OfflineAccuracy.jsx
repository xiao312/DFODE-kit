import React, { useState } from "react";
import { DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const names = {"state-boxcox":"Transformed state", "signed-power":"Signed power", "budget-linear":"State-budget linear", "scaled-asinh":"Scaled asinh", zero:"Zero change"};
const percent = value => value == null ? "Unknown" : `${(100*value).toFixed(2)}%`;
const number = value => value == null ? "Unknown" : value.toLocaleString("en-US", {maximumFractionDigits:2});

export function OfflineAccuracy() {
  const { snapshot, visible } = useDataApp();
  const [seed, setSeed] = useState(20261011);
  const [size, setSize] = useState(10000);
  const [floor, setFloor] = useState(1e-15);
  const [measure, setMeasure] = useState("componentRate");
  const [cost, setCost] = useState("trainingSeconds");
  const models = snapshot.queries.offline_models?.rows;
  if (!models) return null;
  const contract = snapshot.queries.offline_contract.rows[0];
  const audit = snapshot.queries.offline_audit.rows;
  const magnitude = snapshot.queries.offline_magnitudes.rows.filter(row=>row.seed === seed && row.trainingCount === size);
  const magnitudeRows = magnitude.map(row=>({...row, targetName:names[row.target],
    range:`[${row.lower.toExponential(0)}, ${row.upper == null ? "∞" : row.upper.toExponential(0)})`}));
  const curves = snapshot.queries.offline_curves.rows.filter(row => row.atol === floor &&
    (row.target === "zero" || (row.seed === seed && row.trainingCount === size)));
  const curveRows = curves.map(row => ({...row, targetName:names[row.target]})).sort((a,b)=>a.rtol-b.rtol);
  const table = models.map(row => ({...row, targetName:names[row.target], inferenceMs:1000*row.inferenceSeconds/row.states}));
  const maxState = Math.max(...models.map(row=>row.stateRate));
  const minimum = Math.min(...models.map(row=>row.componentRate));
  const maximum = Math.max(...models.map(row=>row.componentRate));
  const source = (id, query, text) => visible(id) && <ReportSection id={id} queryId={query} title={id} showHeading={false} sourceRows={snapshot.queries[query].rows}><RichNarrative id={`${id}:body`} value={text}/></ReportSection>;
  return <section aria-label="Offline representation accuracy">
    {source("offline-opening", "offline_models", `## Current stage: measure representation accuracy before CFD\n\nAll 16 fits completed: four target representations, two nested training sizes, and two fixed seeds. At the primary tolerance, species-component acceptance ranges from ${percent(minimum)} to ${percent(maximum)}. The best observed complete-state acceptance is ${percent(maxState)}; the predeclared research target is 99%. This is a descriptive range over all fits, not a selected deployment model.\n\nA complete state passes only when every non-argon species passes. A high component score can therefore coexist with a low state score. These offline comparisons do not establish CFD accuracy or a guarantee on every prediction.`)}
    {source("offline-contract", "offline_contract", `### 1. Define an acceptable answer\n\nFor a reference increment d, require |prediction − d| ≤ 10⁻¹⁵ + 0.1 × |d|. At d = 10⁻¹², this permits 1.01 × 10⁻¹³ absolute error. The proposed 10⁻¹⁴ error is about 1% and remains a stretch target. This is our research criterion, not a CVODE error guarantee.\n\nPressure now covers 0.95–1.05 atm. Its input uses that declared physical range, not the narrow spread of the old samples. Temperature and composition retain the existing augmentation. Each fit uses four 800-unit GELU layers, FP32, L1 loss, batch size 256, and exactly 2,000 updates. We save the final update, with no evaluation-based checkpoint selection. Target scales use training data only.\n\nCVODE produced ${number(contract.accepted)} accepted training labels and ${number(contract.evaluation_states)} evaluation labels. ${contract.excluded.train} training and ${contract.excluded.validation} evaluation candidates were excluded; no failed label became zero. Label generation took ${number(contract.label_generation_wall_seconds)} wall seconds. The two evaluation snapshots come from the same flame realization as training, not an independent flame case.`)}
    <div style={{display:"flex",gap:"1rem",flexWrap:"wrap",marginBlock:"1rem"}} aria-label="Tolerance plot controls">
      <label>Seed <select value={seed} onChange={event=>setSeed(Number(event.target.value))}><option value={20261011}>20261011</option><option value={20261012}>20261012</option></select></label>
      <label>Training states <select value={size} onChange={event=>setSize(Number(event.target.value))}><option value={2000}>2,000</option><option value={10000}>10,000</option></select></label>
      <label>Absolute floor <select value={floor} onChange={event=>setFloor(Number(event.target.value))}>{[1e-12,1e-15,1e-18].map(value=><option key={value} value={value}>{value.toExponential(0)}</option>)}</select></label>
      <label>Acceptance <select value={measure} onChange={event=>setMeasure(event.target.value)}><option value="componentRate">Species components</option><option value="stateRate">Complete states</option></select></label>
    </div>
    {visible("offline-tolerance") && <EvidenceChart id="offline-tolerance" queryId="offline_curves" title={`Acceptance versus relative tolerance — seed ${seed}, ${number(size)} states, floor ${floor.toExponential(0)}`} rows={curveRows} sourceRows={curves} height={340}
      spec={{type:"line",x:"relativeLabel",y:measure,series:"targetName",stackable:false,valueDecimals:2,xLabel:"Allowed relative error",yLabel:"Acceptance"}}/>}
    {source("offline-next", "offline_contract", `The zero-change baseline passes ${percent(contract.zeroComponentRate)} of components and ${percent(contract.zeroStateRate)} of states at the primary tolerance. Its SSPI is ${percent(contract.zeroSspiRate)} because it predicts no large value where the reference is tiny. Thus, SSPI alone cannot establish useful accuracy. Zero change also passes a 100% relative-error allowance by construction.\n\n### 2. Compare accuracy with cost\n\nThe next plot uses all 16 fits at the primary tolerance. Each point is one target, size, and seed. CPU time includes training preparation and periodic evaluations, but excludes label generation and later verification. Wall time also includes CPU throttling. These are not GPU performance estimates.\n\nThe 2k and 10k samples are nested within each seed. More data at the same update count tests data density, not more optimizer work. The four representations use identical sampled batches and initial weights within each size and seed. Both seeds remain visible; we do not select the better one.`)}
    <label>Cost plot <select value={cost} onChange={event=>setCost(event.target.value)}><option value="trainingSeconds">Training CPU seconds</option><option value="inferenceMs">Inference CPU ms per state</option></select></label>
    {visible("offline-cost") && <EvidenceChart id="offline-cost" queryId="offline_models" title="Acceptance versus cost — primary tolerance, all sizes and seeds" rows={table} sourceRows={models} height={340}
      spec={{type:"scatter",x:cost,y:measure,series:"targetName",stackable:false,valueDecimals:3,xLabel:cost === "trainingSeconds" ? "Training process CPU seconds" : "Inference process CPU ms per state",yLabel:"Acceptance"}}/>}
    {visible("offline-all-models") && <DataComponent id="offline-all-models" queryId="offline_models" kind="table" title="Every fit: primary tolerance and measured cost" sourceRows={models} displayRows={table}>
      <DataTable rows={table} label="Offline matched comparison" compactNumbers={false} columns={[
        {field:"targetName",label:"Target"},{field:"seed",label:"Seed"},{field:"trainingCount",label:"Training states",renderCell:number},
        {field:"componentRate",label:"Components pass",renderCell:percent},{field:"stateRate",label:"States pass",renderCell:percent},
        {field:"sspiRate",label:"SSPI",renderCell:percent},{field:"trainingSeconds",label:"CPU s",renderCell:number},
        {field:"inferenceMs",label:"Inference CPU ms/state",renderCell:value=>value.toPrecision(3)},
      ]}/>
    </DataComponent>}
    {visible("offline-magnitudes") && <DataComponent id="offline-magnitudes" queryId="offline_magnitudes" kind="table" title={`Acceptance by increment magnitude — seed ${seed}, ${number(size)} training states, primary tolerance`} sourceRows={magnitude} displayRows={magnitudeRows}>
      <DataTable rows={magnitudeRows} label="Magnitude-specific acceptance" compactNumbers={false} columns={[
        {field:"targetName",label:"Target"},{field:"range",label:"|Reference increment|"},{field:"components",label:"Components",renderCell:number},
        {field:"pass_fraction",label:"Pass",renderCell:percent},{field:"absolute_error_p99",label:"Absolute error p99",renderCell:value=>value == null ? "No observations" : value.toExponential(2)},
      ]}/>
    </DataComponent>}
    {source("offline-audit", "offline_audit", `### 3. Keep uncertainty separate from model error\n\nThe independent reference check covers ${contract.audited_evaluation_states} evaluation states, not all ${contract.evaluation_states}. It compares tighter and step-limited CVODE with direct-increment Radau solves. A checked component is qualified only when estimated uncertainty is at most 10% of its tolerance budget. Its prediction must then pass with that uncertainty added to the error. This is an empirical margin, not a rigorous bound.\n\nAt the primary tolerance, ${audit[0].qualified_components} of ${audit[0].components} audited components and ${audit[0].qualified_states} of ${audit[0].states} audited states qualify. The remaining ${audit[0].unknown_components} components are unknown at this precision. Full-population plots are nominal scores against CVODE labels, not certified acceptance.\n\nThe next increase in data size must keep the pressure domain, held-out rows, and tolerance grid fixed. Compare the accuracy gain with added label and training cost. If more rows do not help at fixed updates, test a larger update budget separately. Do not add residual stages or return to CFD until the offline error pattern gives a reason.\n\nThe following sections preserve the earlier experiments. Their CFD transfer failure is not a pass/fail gate for this new representation stage.`)}
  </section>;
}
