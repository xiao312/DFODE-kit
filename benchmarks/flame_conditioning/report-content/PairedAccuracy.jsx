import React, {useState} from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {methodLink, recipeNames, targetNames, catalogueUrl} from "./method-catalogue.mjs";

const percent = v => v == null ? "Not evaluated" : `${(100*v).toFixed(2)}%`;
const number = v => v == null ? "Not available" : v.toPrecision(4);
const policies = ["increment-reference-v1", "state-endpoint-v1"];

export function PairedAccuracy() {
  const {snapshot, visible} = useDataApp();
  const [seed, setSeed] = useState(20261011);
  const [policy, setPolicy] = useState(policies[0]);
  const [target, setTarget] = useState("gbct");
  const [floor, setFloor] = useState(1e-15);
  const [measure, setMeasure] = useState("componentRate");
  const models = snapshot.queries.paired_models?.rows;
  if (!models) return null;
  const label = row => ({...row, method:recipeNames[row.name]});
  const selected = models.filter(r=>r.seed === seed && r.policy === policy);
  const range = (name, scoring=policies[0], field="componentRate") => {
    const values = models.filter(r=>r.name === name && r.policy === scoring).map(r=>r[field]);
    return `${percent(Math.min(...values))}–${percent(Math.max(...values))}`;
  };
  const text = (id, value) => visible(id) && <ReportSection id={id} queryId="paired_models" sourceRows={models} title={id} showHeading={false}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  const curves = snapshot.queries.paired_curves.rows.filter(r=>r.seed === seed && r.policy === policy && r.target === target && r.atol === floor && r.role === "paired-grid");
  const zero = snapshot.queries.paired_zero.rows.filter(r=>r.policy === policy && r.atol === floor && r.role === "paired-grid");
  const curveRows = [...curves.map(label), ...zero.map(r=>({...r, method:"Zero-increment control", relativeLabel:`${100*r.rtol}%`, componentRate:r.component_pass_fraction, stateRate:r.state_pass_fraction}))].sort((a,b)=>a.rtol-b.rtol);
  const detail = query => snapshot.queries[query].rows.filter(r=>r.seed === seed && r.policy === policy && r.target === target);
  const audit = detail("paired_audit");
  const binRows = detail("paired_bins").map(r=>({...label(r), range:`[${r.lower.toExponential(0)}, ${r.upper == null ? "∞" : r.upper.toExponential(0)})`, componentRate:r.components ? r.component_pass_count/r.components : null}));
  const app = models.filter(r=>r.policy === policies[1]).map(label);
  return <section aria-label="GBCT and paired error scales">
    {text("paired-opening", `# Does GBCT help, and which error should we control?\n\nWith the unchanged increment pass rule, coordinate training reaches **${range("gbct-coordinate")}** with GBCT and **${range("state-boxcox-coordinate")}** with the transformed-state target. Increment-loss training reaches ${range("gbct-increment")} and ${range("state-boxcox-increment")}, respectively. These ranges retain both seeds; they are not confidence intervals.\n\nThe same GBCT coordinate models reach ${range("gbct-coordinate", policies[1])} when the allowance scales with the final state. That is a different scoring question, **not a gain in the predictions**. No complete state passes the primary increment rule. Independent-test scores remain unavailable.\n\n**The ranking changes at the separately declared tighter state budget** (a=1e-12, r=1e-6): GBCT coordinate training reaches ${range("gbct-coordinate", policies[1], "applicationComponentRate")}, versus ${range("state-boxcox-coordinate", policies[1], "applicationComponentRate")} for the transformed-state control. This is a reason to retain both targets, not proof of a universal winner.\n\nThis is a new, fully GPU-trained matched campaign. Earlier CPU results remain below. Compare methods within this campaign; its software, schedule and timing differ from earlier CPU runs.`)}
    {text("paired-method", `## What we tested, step by step\n\n1. Keep the same 10,000 training states and 1,023 development states. Keep the network, batches, seeds and final-checkpoint rule fixed.\n2. Compare two targets: the transformed-state increment and **GBCT transformed-state rate**. GBCT applies a signed square root after dividing the Box–Cox state difference by the chemistry interval. Both targets use our train-only normalization. This is a target adaptation, not a full reproduction of GBCTNet.\n3. Train each target for 2,000 coordinate-loss updates. Copy the experiment conditions into three arms: another 2,000 coordinate-loss updates, increment-scaled physical-loss updates, or state-scaled physical-loss updates. The code verifies matching warmup weights.\n4. Score every final prediction under both rules. Both use the same error, abs(predicted increment − reference increment). Increment allowance is **a + r × |reference increment|**. State allowance is **a + r × |initial state + reference increment|**. The paired grid uses identical a and r.\n5. Use an RTX 4090 with FP32 network arithmetic, TF32 disabled, and FP64 reconstruction and physical loss. Timings include synchronization. CPU process time is not GPU time.\n\nUse the [canonical method flowcharts](${catalogueUrl}) and [GBCT source check](https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/docs/agents/gbct-source-and-adaptation.md) for exact definitions. Invalid inverse domains are corrected to a zero endpoint and counted; no conservation projection is applied. Only 16 development states have independent reference checks.`)}
    <div style={{display:"flex",flexWrap:"wrap",gap:"1rem",marginBlock:"1rem"}} aria-label="Paired comparison controls">
      <label>Seed <select value={seed} onChange={e=>setSeed(Number(e.target.value))}>{[20261011,20261012].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{policies.map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Target for tolerance and detail views <select value={target} onChange={e=>setTarget(e.target.value)}>{["state-boxcox","gbct"].map(v=><option key={v} value={v}>{targetNames[v]}</option>)}</select></label>
      <label>Absolute floor <select value={floor} onChange={e=>setFloor(Number(e.target.value))}>{[1e-12,1e-15,1e-18].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Acceptance <select value={measure} onChange={e=>setMeasure(e.target.value)}><option value="componentRate">Species components</option><option value="stateRate">Complete states</option></select></label>
    </div>
    {visible("paired-tolerance") && <EvidenceChart id="paired-tolerance" queryId="paired_curves" queryIds={["paired_curves","paired_zero"]} sourceRowsByQuery={{paired_curves:curves,paired_zero:zero}} title={`Acceptance versus tolerance — ${targetNames[target]}, ${policy}, seed ${seed}`} rows={curveRows} height={360} spec={{type:"line",x:"relativeLabel",y:measure,series:"method",stackable:false,valueDecimals:2,xLabel:`Relative allowance; absolute floor ${floor}`,yLabel:"Acceptance"}}/>}
    {visible("paired-cost") && <EvidenceChart id="paired-cost" queryId="paired_models" sourceRows={selected} title={`Acceptance versus GPU fit time — both targets, ${policy}, seed ${seed}`} rows={selected.map(label)} height={330} spec={{type:"scatter",x:"trainingSeconds",y:measure,series:"objective",stackable:false,valueDecimals:3,xLabel:"Synchronized fit wall seconds",yLabel:"Acceptance"}}/>}
    {visible("paired-table") && <DataComponent id="paired-table" queryId="paired_models" kind="table" title={`Both targets and both seeds — ${policy}, a=1e-15, r=0.1`} sourceRows={models.filter(r=>r.policy === policy)} displayRows={models.filter(r=>r.policy === policy).map(label)}><DataTable rows={models.filter(r=>r.policy === policy).map(label)} label="Paired train and development scores" columns={[
      {field:"method",label:"Method",renderCell:(_,r)=><a href={methodLink(r.name)}>{recipeNames[r.name]}</a>},{field:"seed",label:"Seed"},
      {field:"trainingComponentRate",label:"Train components",renderCell:percent},{field:"componentRate",label:"Development components",renderCell:percent},
      {field:"trainingStateRate",label:"Train complete states",renderCell:percent},{field:"stateRate",label:"Development complete states",renderCell:percent},
      {field:"testRate",label:"Independent test",renderCell:percent},{field:"normalizedP99",label:"Error / allowance p99",renderCell:number},
      {field:"trainingSeconds",label:"GPU fit wall s",renderCell:number},{field:"inferenceMs",label:"Batched end-to-end ms/state",renderCell:number},
      {field:"correctionRate",label:"Inverse corrections",renderCell:percent},{field:"negativeRate",label:"Negative endpoints",renderCell:percent},{field:"massDriftP99",label:"Mass drift p99",renderCell:number},
    ]}/></DataComponent>}
    {visible("paired-application") && <DataComponent id="paired-application" queryId="paired_models" kind="table" title="Separate application-state diagnostic: a=1e-12, r=1e-6" sourceRows={models.filter(r=>r.policy === policies[1])} displayRows={app}><DataTable rows={app} label="Stricter state-relative allowance" columns={[
      {field:"method",label:"Method"},{field:"seed",label:"Seed"},{field:"applicationComponentRate",label:"Development components",renderCell:percent},{field:"applicationStateRate",label:"Development complete states",renderCell:percent},
    ]}/></DataComponent>}
    {visible("paired-audit") && <DataComponent id="paired-audit" queryId="paired_audit" kind="table" title={`Reference qualification — ${policy}, 16 states`} sourceRows={audit} displayRows={audit.map(label)}><DataTable rows={audit.map(label)} label="Empirical reference qualification" columns={[
      {field:"method",label:"Method"},{field:"components",label:"Qualified / 928 components"},{field:"passed_components",label:"Qualified passes"},{field:"states",label:"Qualified / 16 states"},{field:"passed_states",label:"Qualified state passes"},
    ]}/></DataComponent>}
    {visible("paired-bins") && <DataComponent id="paired-bins" queryId="paired_bins" kind="table" title={`Acceptance by increment size — ${policy}`} sourceRows={detail("paired_bins")} displayRows={binRows}><DataTable rows={binRows} label="Magnitude-bin acceptance" columns={[
      {field:"method",label:"Method"},{field:"range",label:"|Reference increment|"},{field:"components",label:"Components"},{field:"componentRate",label:"Pass",renderCell:percent},
    ]}/></DataComponent>}
    {visible("paired-species") && <DataComponent id="paired-species" queryId="paired_species" kind="table" title={`Species errors — ${policy}`} sourceRows={detail("paired_species")} displayRows={detail("paired_species").map(label)}><DataTable rows={detail("paired_species").map(label)} label="Species acceptance and error tails" columns={[
      {field:"method",label:"Method"},{field:"species",label:"Species"},{field:"componentRate",label:"Pass",renderCell:percent}, ...["p50","p95","p99","max"].map(field=>({field,label:`Error / allowance ${field}`,renderCell:number})),
    ]}/></DataComponent>}
    {text("paired-next", "## What this result can decide\n\nCompare targets within the same training objective and scoring rule. Compare objectives within the same target. A change in the scoring rule alone is not an accuracy improvement. The separate application-state table uses a different, tighter relative allowance than the paired 10% rule; do not merge its scores with the primary benchmark.\n\nThe next controlled factor is dataset size, with the selected recipes and sufficient training work held explicit. A fresh independent-case test comes only after the recipe is fixed. These one-step development scores do not establish trajectory accuracy, saturation, or solver-level reliability.")}
  </section>;
}
