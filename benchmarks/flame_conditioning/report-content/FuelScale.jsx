import React, {useState} from "react";
import {DataComponent, DataTable, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {recipeNames, methodLink} from "./method-catalogue.mjs";
const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;
const compact = value => new Intl.NumberFormat("en", {notation:"compact",maximumFractionDigits:1}).format(value);
const decimal = value => value.toPrecision(4);

export function FuelScale() {
  const {snapshot, visible} = useDataApp();
  const [policy, setPolicy] = useState("increment-reference-v1");
  const models = snapshot.queries.fuel_scale_models?.rows;
  if (!models) return null;
  const latest = Math.max(...models.map(r=>r.trainingCount));
  const selected = models.filter(r=>r.policy === policy);
  const display = selected.map(r=>({...r,method:recipeNames[r.recipe]}));
  const throughput = snapshot.queries.fuel_label_throughput.rows;
  const range = (recipe, size) => {
    const values = models.filter(r=>r.policy === "increment-reference-v1" && r.recipe === recipe && r.trainingCount === size).map(r=>r.developmentRate);
    return `${percent(Math.min(...values))}–${percent(Math.max(...values))}`;
  };
  const meanRate = workers => {
    const values = throughput.filter(r=>r.workers === workers).map(r=>r.rows_per_second);
    return values.reduce((a,b)=>a+b,0)/values.length;
  };
  const text = (id, value, queryId="fuel_scale_models") => visible(id) && <ReportSection id={id} queryId={queryId} title={id} showHeading={false} sourceRowsByQuery={{[queryId]:snapshot.queries[queryId].rows}}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  return <section aria-label="Dataset size and reference throughput">
    {text("fuel-scale-opening", `# Larger Fuel-recipe runs: what improved, and what changed\n\n## Executive Summary\n\n- At **${compact(latest)} training rows**, Fuel state-target development acceptance is **${range("fuel-state",latest)}**, compared with ${range("fuel-state",10000)} at 10k. Direct-power acceptance is **${range("fuel-power",latest)}**, compared with ${range("fuel-power",10000)}. These ranges retain both seeds, not only the best run.\n- The primary increment pass rule remains **abs(error) ≤ 1e-15 + 0.1 × abs(reference increment)**. ${models.filter(r=>r.policy === "increment-reference-v1" && r.trainingCount === latest).every(r=>r.stateRate === 0) ? "No complete development state passes this rule yet." : "Complete-state acceptance is reported separately in the table."}\n- These are **larger-data, larger-work runs**. The network and epoch schedules are fixed, but the number of batches and weight updates increases. We cannot assign all improvement to dataset size alone. The independent test remains unopened for this campaign.`)}
    {text("fuel-scale-terms", `## Read the work counts correctly\n\n1. A **row** is one starting temperature, pressure and composition, plus reference species changes over 1 microsecond. The model takes 61 input values and predicts 58 non-argon increments. It is not a whole trajectory.\n2. A **batch** groups rows for one loss and gradient calculation. An **update** changes the network weights once. It is not a CVODE time step.\n3. An **epoch** passes through the training pool. The state recipe drops the last incomplete batch; the power recipe keeps it.\n4. **Row presentations** count repeated uses of rows. They are not new examples. At 10k, one batch covers the pool. At 50k, the state recipe makes 2 updates per epoch; power makes 3. At 200k, both make 10.\n\nEach larger selection contains the preceding saved training selection. All sizes use the same 1,023 development states and labels. Training and development still come from different snapshots of one flame, not independent flame cases.`)}
    <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{["increment-reference-v1","state-endpoint-v1"].map(value=><option key={value}>{value}</option>)}</select></label>
    {visible("fuel-scale-results") && <DataComponent id="fuel-scale-results" queryId="fuel_scale_models" kind="table" title={`Completed size comparison — ${policy}, a=1e-15, r=0.1`} sourceRows={selected} displayRows={display}><DataTable rows={display} label="Every completed Fuel size, recipe and seed" columns={[
      {field:"trainingCount",label:"Unique training rows",renderCell:compact},
      {field:"method",label:"Recipe",renderCell:(_,row)=><a href={methodLink(row.recipe)}>{row.method}</a>},
      {field:"seed",label:"Seed"},{field:"trainingRate",label:"Training pass",renderCell:percent},
      {field:"developmentRate",label:"Development pass",renderCell:percent},{field:"stateRate",label:"Complete states",renderCell:percent},
      {field:"updates",label:"Weight updates",renderCell:compact},{field:"presentations",label:"Row presentations",renderCell:compact},
      {field:"fitSeconds",label:"GPU fit wall s",renderCell:decimal},{field:"p99",label:"Error / allowance p99",renderCell:decimal},
      {field:"negativeRate",label:"Negative endpoints",renderCell:percent},{field:"correctionRate",label:"Inverse corrections",renderCell:percent},
      {field:"independentTest",label:"Independent test",renderCell:percent},
    ]}/></DataComponent>}
    {text("fuel-throughput-note", `## Reference labels can be generated faster without weaker tolerances\n\nEight CPU workers produced about **${Math.round(meanRate(8))} rows/s**, compared with **${Math.round(meanRate(1))} rows/s** for one worker: about **${(meanRate(8)/meanRate(1)).toFixed(1)}×** faster on this 2,048-row benchmark. Both repetitions matched all saved serial increments and acceptance flags exactly. Startup and result transport are included. Full-run sampling, chunk I/O and final export are not included in this rate.\n\nThe new runner saves small completed chunks and can resume into a new directory. It reuses checked labels and leaves earlier results unchanged. Each new dataset still needs a fresh CVODE/Radau subset audit before GPU training. More rows from the same source snapshots increase sample density, not independent case coverage. The next size gate is 1M only after reviewing completed 200k results and source coverage; it is not a reported result here.`, "fuel_label_throughput")}
    {visible("fuel-throughput") && <DataComponent id="fuel-throughput" queryId="fuel_label_throughput" kind="table" title="CPU reference-label benchmark — exact serial parity" sourceRows={throughput}><DataTable rows={throughput} searchable={false} label="Both worker benchmark repetitions" columns={[
      {field:"workers",label:"Workers"},{field:"repetition",label:"Repeat (0-based)"},{field:"rows",label:"Rows"},
      {field:"seconds",label:"Wall seconds",renderCell:decimal},{field:"rows_per_second",label:"Rows / second",renderCell:decimal},
      {field:"exact_serial_labels",label:"Exact increments",renderCell:v=>v ? "Pass" : "Fail"},
      {field:"exact_acceptance_flags",label:"Exact flags",renderCell:v=>v ? "Pass" : "Fail"},
    ]}/></DataComponent>}
  </section>;
}
