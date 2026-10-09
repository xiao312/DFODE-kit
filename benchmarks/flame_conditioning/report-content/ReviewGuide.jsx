import React from "react";
import {DataComponent, DataTable, ReportSection, RichNarrative, useDataApp} from "../../data-app-public.jsx";
import {catalogueUrl, methodLink, recipeNames} from "./method-catalogue.mjs";

const percent = value => value == null ? "Not evaluated" : `${(100*value).toFixed(2)}%`;

export function ReviewGuide() {
  const {snapshot, visible} = useDataApp();
  const models = snapshot.queries.improve_models?.rows;
  if (!models) return null;
  const bases = snapshot.queries.refinement_models.rows.filter(row=>row.name === "long-state-boxcox");
  const sourceRowsByQuery = {improve_models:models, refinement_models:bases};
  const rows = [...bases,...models].map(row=>({...row, method:recipeNames[row.name], testRate:null}));
  const text = (id, value) => visible(id) && <ReportSection id={id} queryId="improve_models" queryIds={["improve_models","refinement_models"]} sourceRowsByQuery={sourceRowsByQuery} title={id} showHeading={false}><RichNarrative id={`${id}:body`} value={value}/></ReportSection>;
  return <section aria-label="Review guide: splits, methods and tolerances">
    {text("review-split-guide", `## How to read the earlier CPU results\n\n**Training** means the 10,000 selected states used to fit each run. **Development/validation** means 1,023 states not used for gradient fitting, but inspected repeatedly to choose later methods. **Independent test** means a separate case reserved until the recipe and scoring rule are fixed. We have not run that test for these current models.\n\nThe training and development states come from different snapshots of one flame realization: four training snapshots and two development snapshots, split before augmentation. This is not an independent-case test. Earlier CFD test tables below concern older models and a different experiment; do not use those scores as the test column for this campaign.\n\nThe table shows final-checkpoint scores for both seeds. All complete-state scores in this earlier increment-scored table are zero. Local-table training scores query points present in the table, not leave-one-out points. Their 100% training score and poor development score show why the splits must be separate. The neural training scores are also far from complete acceptance; these runs do not establish saturation.\n\nUse the [method catalogue with flowcharts](${catalogueUrl}) for names, formulas, steps, aliases and source code. A representation is only the output coordinate. A recipe also specifies inputs, architecture, loss and optimization. A run adds data size, seed and work budget.`)}
    {visible("review-split-scores") && <DataComponent id="review-split-scores" queryId="improve_models" queryIds={["improve_models","refinement_models"]} sourceRowsByQuery={sourceRowsByQuery} kind="table" title="Train, development and independent test — fixed primary tolerance" displayRows={rows}>
      <DataTable rows={rows} label="Final checkpoint split scores" columns={[
        {field:"method",label:"Canonical method",renderCell:(_,row)=><a href={methodLink(row.name)}>{recipeNames[row.name]}</a>},
        {field:"seed",label:"Seed"},
        {field:"trainingComponentRate",label:"Train components",renderCell:percent},
        {field:"componentRate",label:"Development components",renderCell:percent},
        {field:"trainingStateRate",label:"Train complete states",renderCell:percent},
        {field:"stateRate",label:"Development complete states",renderCell:percent},
        {field:"testRate",label:"Independent test",renderCell:percent},
      ]}/>
    </DataComponent>}
    {text("review-tolerance-guide", "### What do the tolerance terms mean?\n\nThe historical increment rule remains **allowed absolute error = 1e-15 + 0.1 × |reference increment|**. It permits 1.01e-13 at an increment of 1e-12, 2e-15 at 1e-14, and 1.01e-15 at 1e-16. Below the crossover of 1e-14, the absolute floor dominates. A zero prediction passes for |increment| ≤ approximately 1.11e-15. Passing does not establish accurate significant digits for extremely small changes.\n\nIn standard CVODE terminology, atol and rtol are fixed parameters. The **relative contribution**, rtol × |state|, and the **total allowance**, atol + rtol × |state|, vary with the state. CVODE uses an estimated local error and a weighted RMS test; our every-species surrogate acceptance is a different test.\n\nThe paired campaign above evaluates both increment-reference and state-endpoint scaling. It uses the reference endpoint Y + d, not the predicted endpoint. Both numerators use the saved signed increment error, so endpoint rounding cannot hide it. The full grid and zero control keep the effect of a changed scoring rule visible.\n\nA separate custom policy can vary **both** parameters: tau_i(m) = a_i(m) + r_i(m) × m. This is not standard CVODE automatic behavior. No numerical curve for this policy has been selected or tested. Define its total allowance, units and zero behavior from the required accuracy before evaluating it; do not tune the allowance to make existing model errors pass.\n\nOnly 16 development states have independent reference-uncertainty checks. Full-population rates remain nominal. Unknown reference accuracy is not a pass. Neither one-step metric establishes trajectory accuracy. See the [canonical accuracy protocol](https://github.com/xiao312/DFODE-kit/blob/research/precision-conditioned-increments/docs/agents/representation-accuracy-success-sources.md#required-dual-scaling-protocol-for-later-runs) for the fixed definitions.")}
  </section>;
}
