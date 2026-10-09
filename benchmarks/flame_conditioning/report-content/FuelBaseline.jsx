import React, {useState} from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {recipeNames, methodLink, catalogueUrl} from "./method-catalogue.mjs";
const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;
const number = value => value == null ? "Not available" : value.toPrecision(4);

export function FuelBaseline() {
  const {snapshot, visible} = useDataApp();
  const [policy, setPolicy] = useState("increment-reference-v1");
  const [seed, setSeed] = useState(20261011);
  const [recipe, setRecipe] = useState("fuel-state");
  const models = snapshot.queries.fuel_models?.rows;
  if (!models) return null;
  const source = models.filter(r=>r.policy === policy);
  const rows = source.map(r=>({...r, method:recipeNames[r.recipe]}));
  const history = snapshot.queries.fuel_history.rows.filter(r=>r.policy === policy && r.seed === seed && r.recipe === recipe);
  const range = id => {
    const values = models.filter(r=>r.policy === "increment-reference-v1" && r.recipe === id).map(r=>r.developmentRate);
    return `${percent(Math.min(...values))}–${percent(Math.max(...values))}`;
  };
  const old = snapshot.queries.paired_models.rows.filter(r=>r.name === "state-boxcox-coordinate" && r.policy === "increment-reference-v1");
  const text = (id, value) => visible(id) && <ReportSection id={id} queryId="fuel_models" queryIds={id === "fuel-opening" ? ["fuel_models","paired_models"] : ["fuel_models"]} sourceRowsByQuery={id === "fuel-opening" ? {fuel_models:models,paired_models:old} : {fuel_models:models}} title={id} showHeading={false}><RichNarrative id={`${id}:body`} value={value} sourcePreviews={{
    "https://arxiv.org/html/2507.08277v2": {title:"Fuel study: physics-aware augmentation and scale separation",summary:"The author manuscript defines flame-based data preparation, two target transformations, and temperature-dependent deployment.",approvedForReport:true},
    "https://arxiv.org/html/2512.05685v1": {title:"GBCT: an output scaling layer for multiscale ODEs",summary:"The author manuscript specifies a transformed-state rate target and a two-stage large-batch training schedule.",approvedForReport:true},
  }}/></ReportSection>;
  return <section aria-label="Fuel source-recipe baseline">
    {text("fuel-opening", `# The source training recipes need more than matching epoch counts\n\nThe first **reduced-data Fuel recipe replication did not improve acceptance**. On the same 10k training rows, transformed-state development acceptance is **${range("fuel-state")}**, versus 33.49–33.69% for the prior 4k-update coordinate control. Direct-power acceptance is ${range("fuel-power")}. Both seeds are retained. No complete state passes the primary increment rule.\n\nThe recovered state training log used 7.6M training rows, 20k batches, and 1.5k epochs: **570k updates and 11.4B row presentations**. With only 10k rows, our capped batch covers the entire training set. The same epoch count gives only 1.5k updates and 15M presentations. Copying the epoch count does not reproduce the training budget.\n\nThis result does not refute the Fuel paper. Our data are much smaller, our development split is different, and our error contract is not its paper metric.`)}
    {text("fuel-method", `## What changed, and what stayed fixed\n\n1. Read the [Fuel manuscript](https://arxiv.org/html/2507.08277v2), the [GBCT manuscript](https://arxiv.org/html/2512.05685v1), and the saved Fuel training scripts. The [source audit](https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/docs/agents/fuel-baseline-replication.md) lists verified settings and unresolved data lineage.\n2. Match the Fuel network: 61 inputs, four 800-unit GELU layers, and 58 outputs. Match each script's normalization, L1 loss, epoch schedule, and Adam reset policy. Direct power has **zero target center and a standard-deviation scale**, not centered Z-score or RMS scaling.\n3. Keep the same training row IDs, development rows, references, seeds, and acceptance rules. Keep train-only statistics, stable FP64 reconstruction, and visible invalid-domain corrections. Do not copy validation leakage or silent clipping.\n4. Run all four fits on an RTX 4090 in the project GPU environment. Reload saved weights and check exact predictions, independent acceptance counts, and physical diagnostics.\n\nThe [canonical method catalogue](${catalogueUrl}) contains both recipe flowcharts. GBCT has separate paper/code discrepancies; no unchanged upstream training command is treated as a verified reproduction.`)}
    <div style={{display:"flex",flexWrap:"wrap",gap:"1rem",marginBlock:"1rem"}} aria-label="Fuel result controls">
      <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{["increment-reference-v1","state-endpoint-v1"].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve seed <select value={seed} onChange={e=>setSeed(Number(e.target.value))}>{[20261011,20261012].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve recipe <select value={recipe} onChange={e=>setRecipe(e.target.value)}>{["fuel-state","fuel-power"].map(v=><option key={v} value={v}>{recipeNames[v]}</option>)}</select></label>
    </div>
    {visible("fuel-results") && <DataComponent id="fuel-results" queryId="fuel_models" kind="table" title={`Final 10k fits — ${policy}, a=1e-15, r=0.1`} sourceRows={source} displayRows={rows}><DataTable rows={rows} label="Fuel train and development results" columns={[
      {field:"method",label:"Recipe",renderCell:(_,r)=><a href={methodLink(r.recipe)}>{r.method}</a>},{field:"seed",label:"Seed"},
      {field:"trainingRate",label:"Training pass",renderCell:percent},{field:"developmentRate",label:"Development pass",renderCell:percent},
      {field:"stateRate",label:"Complete states",renderCell:percent},{field:"p99",label:"Error / allowance p99",renderCell:number},
      {field:"sspi",label:"SSPI, tiny subset only",renderCell:percent},{field:"epochs",label:"Epochs"},{field:"updates",label:"Updates"},
      {field:"presentations",label:"Row presentations"},{field:"fitSeconds",label:"GPU fit wall s",renderCell:number},
      {field:"correctionRate",label:"Inverse corrections",renderCell:percent},{field:"negativeRate",label:"Negative endpoints",renderCell:percent},
      {field:"independentTest",label:"Independent test",renderCell:percent},
    ]}/></DataComponent>}
    {visible("fuel-history") && <EvidenceChart id="fuel-history" queryId="fuel_history" title={`Training and development acceptance — ${recipeNames[recipe]}, seed ${seed}`} rows={history} sourceRows={history} height={340} spec={{type:"line",x:"epoch",y:"acceptanceRate",series:"split",stackable:false,xLabel:"Completed epochs (one update per epoch at 10k)",yLabel:"Acceptance",valueDecimals:2}}/>}
    {text("fuel-next", snapshot.queries.fuel_scale_models ? "## This section preserves the original 10k evidence\n\nThe larger completed runs and their work counts are in the size-comparison section above. The 10k table and curves remain unchanged. All expansions keep the same pressure range, source groups, reference tolerances and development rows. CPU is used for numerical references; neural fits require GPU. No expansion here reproduces the paper's full source coverage or eight-million-state dataset." : "## Next gate: increase data without changing the development cases\n\nThis section reports completed 10k results only. Larger campaigns must pass a new reference audit and preserve the original development states and labels before their results are added.")}
  </section>;
}
