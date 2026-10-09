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
    {text("review-split-guide", `## How to read these results\n\n**Training** means the 10,000 selected states used to fit each run. **Development/validation** means 1,023 states not used for gradient fitting, but inspected repeatedly to choose later methods. **Independent test** means a separate case reserved until the recipe and scoring rule are fixed. We have not run that test for these current models.\n\nThe training and development states come from different snapshots of one flame realization: four training snapshots and two development snapshots, split before augmentation. This is not an independent-case test. Earlier CFD test tables below concern older models and a different experiment; do not use those scores as the test column for this campaign.\n\nThe table shows final-checkpoint scores for both seeds. All complete-state development scores are zero. Local-table training scores query points present in the table, not leave-one-out points. Their 100% training score and poor development score show why the splits must be separate. The neural training scores are also far from complete acceptance; these runs do not establish saturation.\n\nUse the [method catalogue with flowcharts](${catalogueUrl}) for names, formulas, steps, aliases and source code. A representation is only the output coordinate. A recipe also specifies inputs, architecture, loss and optimization. A run adds data size, seed and work budget.`)}
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
    {text("review-tolerance-guide", `### Is the current pass rule the best one?\n\nNo optimum has been established. The current benchmark uses **allowed absolute error = 1e-15 + 0.1 × |reference increment|**. It already changes with magnitude: about 10% for large increments, but an absolute floor for tiny increments. At 1e-12 it allows 1.01e-13 (10.1%); at 1e-14 it allows 2e-15 (20%); at 1e-16 it allows 1.01e-15 (1,010%). At zero there is no defined relative error.\n\nThe crossover is 1e-14. Below that, the floor dominates. A zero prediction passes whenever |reference increment| ≤ approximately 1.11e-15. Thus this score does not measure significant digits for extremely small increments. It is an initial research target, not solver-level precision.\n\n**Recommended next scoring design, not a change to these results:** keep the fixed benchmark for continuity, and show the existing grid of absolute floors (1e-12, 1e-15, 1e-18) and relative tolerances (100%, 10%, 1%, 0.1%). Keep magnitude bins and the zero predictor beside the totals. This makes a change in the pass rule visible rather than presenting it as a model improvement.\n\nA more flexible budget can be written tau_i(m) = a_i + r_i(m) × m, where m = |reference increment|. Prefer a species-specific absolute floor and, only with a physical reason, a smooth magnitude-dependent relative allowance. Set those choices from the required chemistry accuracy and independently checked reference resolution, before the next test. Do not fit them to observed model errors, inflate them to hide uncertain labels, or use the prediction magnitude for scoring. Check that the resulting absolute allowance is nondecreasing with m.\n\nReport state-update accuracy separately, using a species budget based on the initial and final state. [CVODE uses state-based weights and a weighted RMS error test](https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html), not our every-species increment pass rule. Neither one-step score alone guarantees trajectory accuracy.\n\nOnly 16 development states have independent reference-uncertainty checks. Full-population rates remain nominal. If reference uncertainty is too large for a requested budget, mark that score unqualified instead of relaxing the budget. No model was retrained or newly tested for this review update.`)}
  </section>;
}
