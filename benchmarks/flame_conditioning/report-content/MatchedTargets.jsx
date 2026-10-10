import React, {useState} from "react";
import {DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {targetNames, methodLink} from "./method-catalogue.mjs";

const targets = ["state-boxcox", "signed-power", "scaled-asinh", "gbct"];
const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;
const numeric = value => value.toPrecision(4);

export function MatchedTargets() {
  const {snapshot, visible} = useDataApp();
  const [policy, setPolicy] = useState("increment-reference-v1");
  const [seed, setSeed] = useState(20261011);
  const [target, setTarget] = useState("gbct");
  const [axis, setAxis] = useState("updates");
  const [atol, setAtol] = useState(1e-15);
  const all = snapshot.queries.matched_targets_models?.rows;
  if (!all) return null;
  const selected = all.filter(r=>r.policy === policy);
  const history = snapshot.queries.matched_targets_history.rows.filter(r=>r.policy === policy && r.seed === seed && r.target === target);
  const curves = snapshot.queries.matched_targets_curves.rows.filter(r=>r.policy === policy && r.seed === seed && r.target === target && r.atol === atol && r.role === "paired-grid");
  const narrative = (id,text) => visible(id) && <ReportSection id={id} queryId="matched_targets_models" sourceRows={all} title={id} showHeading={false}><RichNarrative id={`${id}:body`} value={text}/></ReportSection>;
  return <section aria-label="Frozen matched-target baseline">
    {narrative("matched-opening", `# GBCT plus increment-scaled loss is the strongest acceptance baseline\n\n## Executive Summary\n\nThe complete **24-fit comparison** is now the frozen development baseline. GBCT plus increment-scaled loss passes **66.29–66.55%** of species components in the two seeds, versus **62.88–63.11%** for the transformed-state coordinate control. Its training acceptance is **69.92–70.04%**.\n\nGBCT's increment-scaled loss improves acceptance and reduces its error/allowance p99 from **59–74 to 23–27**. But **no complete development state passes**. The GBCT outputs have no negative endpoints after their declared inverse-domain corrections; those corrections occur in **0.21–0.26%** of species components.\n\n“Baseline” means a fixed model comparison, not a trusted numerical reference or a solver-ready model. The independent test remains unopened. The reference labels and all scoring rules remain unchanged.`)}
    {narrative("matched-method", `## What we compared, step by step\n\n1. Keep the same **200k training rows** per seed and **1,023 development states**, with 58 scored species.\n2. Fit each target's mean and sample standard deviation on the same 50k training prefix. All targets use identical input scales.\n3. Train the same 4×800 GELU network for **18k updates**, batch 10k. Use one common three-stage learning-rate schedule and Adam reset rule.\n4. Use coordinate L1 for the first 12k updates. For the final 6k, keep coordinate L1 or switch to mean log1p of physical error divided by the selected allowance. Paired warmup weights and diagnostics match exactly.\n5. Keep the final checkpoint and both seeds. Verify saved-model replay, acceptance counts, physical metrics and historical control parity.\n\nThe primary rule is **abs(error) ≤ 1e-15 + 0.1 × abs(reference increment)**. The state policy instead uses **abs(reference endpoint)** in the allowance. Both score the same signed-increment error. These policies are different accuracy objectives, not two names for the same score.\n\nGBCT and power use the common schedule and normalization here. They are controlled target adaptations, not the original paper recipes. In particular, the new power result cannot be credited to a new transform relative to the earlier Fuel-power recipe.`)}
    <div style={{display:"flex",flexWrap:"wrap",gap:"1rem",marginBlock:"1rem"}} aria-label="Matched-target controls">
      <label>Error scale <select value={policy} onChange={e=>setPolicy(e.target.value)}>{["increment-reference-v1","state-endpoint-v1"].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve seed <select value={seed} onChange={e=>setSeed(Number(e.target.value))}>{[20261011,20261012].map(v=><option key={v}>{v}</option>)}</select></label>
      <label>Curve target <select value={target} onChange={e=>setTarget(e.target.value)}>{targets.map(v=><option key={v} value={v}>{targetNames[v]}</option>)}</select></label>
      <label>Learning-curve x-axis <select value={axis} onChange={e=>setAxis(e.target.value)}><option value="updates">Updates</option><option value="seconds">Fit wall seconds</option></select></label>
      <label>Tolerance-curve absolute floor <select value={atol} onChange={e=>setAtol(Number(e.target.value))}>{[...new Set(snapshot.queries.matched_targets_curves.rows.filter(r=>r.role === "paired-grid").map(r=>r.atol))].map(v=><option key={v} value={v}>{v.toExponential()}</option>)}</select></label>
    </div>
    {visible("matched-models") && <DataComponent id="matched-models" queryId="matched_targets_models" kind="table" title={`All 24 final fits — ${policy}`} sourceRows={selected} displayRows={selected}><DataTable rows={selected} label="Frozen factorial baseline" columns={[
      {field:"target",label:"Target",renderCell:(_,r)=><a href={methodLink(r.recipe)}>{targetNames[r.target]}</a>},
      {field:"objective",label:"Final-stage loss"},{field:"seed",label:"Seed"},
      {field:"trainingRate",label:"Training pass",renderCell:percent},{field:"developmentRate",label:"Development pass",renderCell:percent},
      {field:"stateRate",label:"Whole-state pass",renderCell:percent},{field:"p99",label:"Error/allowance p99",renderCell:numeric},
      {field:"negativeRate",label:"Negative endpoints",renderCell:percent},{field:"correctionRate",label:"Inverse corrections",renderCell:percent},
      {field:"fitSeconds",label:"GPU fit wall s",renderCell:numeric},{field:"inferenceMicroseconds",label:"Batched inference µs/state",renderCell:numeric},
      {field:"zeroRate",label:"Zero-update pass",renderCell:percent},{field:"independentTest",label:"Independent test",renderCell:percent},
    ]}/></DataComponent>}
    {visible("matched-history") && <EvidenceChart id="matched-history" queryId="matched_targets_history" title={`Learning curves — ${targetNames[target]}, seed ${seed}`} rows={history} sourceRows={history} height={360} spec={{type:"line",x:axis,y:"acceptanceRate",series:"series",stackable:false,valueDecimals:2,xLabel:axis === "updates" ? "Completed updates" : "Fit wall seconds, including diagnostics",yLabel:"Species acceptance"}}/>}
    {visible("matched-tolerance") && <EvidenceChart id="matched-tolerance" queryId="matched_targets_curves" title={`Acceptance versus relative allowance — ${targetNames[target]}, absolute floor ${atol.toExponential()}`} rows={curves} sourceRows={curves} height={320} spec={{type:"line",x:"rtol",y:"acceptanceRate",series:"objective",stackable:false,valueDecimals:2,xLabel:"Relative allowance coefficient",yLabel:"Species acceptance"}}/>}
    {narrative("matched-next", `## Next: learn a correction around the frozen GBCT prediction\n\nThe evidence supports keeping **GBCT with increment-scaled loss** as the starting model for both seeds. It does not establish that one scalar transform solves the accuracy problem. State-scaled training improves state-scaled acceptance while reducing increment-scaled acceptance; the objective matters.\n\nThe adaptive follow-up will predict the remaining physical error, with correction scales calibrated from training residuals. A fixed-scale correction and additional training of the base are required controls. The accuracy allowance stays fixed: changing a representation scale must not make the pass rule easier.\n\nThe development milestone is **at least +5 percentage points in both seeds**, with lower p99 error and no increase in negative endpoints. This is a chosen research target, not an expected or guaranteed gain. A scale-specific claim also requires beating the matched uncalibrated correction. Measure total base-plus-correction inference cost and retain pass-to-fail counts.\n\nThese are repeatedly used, same-flame development states. Only a later frozen independent evaluation can support a generalization claim. All earlier experiments remain below.`)}
  </section>;
}
