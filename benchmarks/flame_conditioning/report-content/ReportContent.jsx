import React from "react";
import { policyBinTable } from "./policy-bins.mjs";
import { largestPrimaryComparison } from "./validation-selection.mjs";
import { DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";

const paper = "https://arxiv.org/html/2507.08277v2";
const names = {"state-boxcox":"Transformed state (Box–Cox)", "signed-power":"Direct signed power", "budget-linear":"Budget-linear", "scaled-asinh":"Scaled asinh"};
const percent = value => `${value !== 0 && Math.abs(value) < 0.0001 ? (100 * value).toPrecision(3) : (100 * value).toFixed(2)}%`;
const integer = value => Math.round(value).toLocaleString("en-US");
const budgetNumber = value => value == null ? "Not recorded" : value !== 0 && Math.abs(value) < 100 ? value.toPrecision(3) : integer(value);
const speciesColumns = ["NH3", "CH4", "NO", "OH"].map(name => ({field:`${name}BudgetP99`,label:`${name} budget p99 ↓`,renderCell:budgetNumber}));

export function ReportContent() {
  const { snapshot, visible, appTitle, canEdit, mode, setAppTitle } = useDataApp();
  const models = snapshot.queries.models.rows;
  const reference = snapshot.queries.reference.rows;
  const cfd = snapshot.queries.cfd.rows;
  const parity = snapshot.queries.runtime_parity?.rows;
  const tolerance = snapshot.queries.cfd_tolerance?.rows;
  const profile = snapshot.queries.flame_profile?.rows;
  const historical = snapshot.queries.historical?.rows;
  const datasets = snapshot.queries.datasets?.rows;
  const expanded = snapshot.queries.expanded_reference?.rows;
  const filterAudit = snapshot.queries.filter_audit?.rows;
  const heldout = snapshot.queries.heldout?.rows;
  const testSampling = snapshot.queries.heldout_sampling?.rows;
  const testReference = snapshot.queries.heldout_reference?.rows;
  const testRuns = [...new Set((heldout || []).filter(row => row.name.startsWith("training-backbone-longer")).map(row => row.name.split("--")[0]))];
  const testRunSize = name => Number(name.match(/longer(\d+)k/)?.[1] || 0);
  const largestTestRun = testRuns.sort((a, b) => testRunSize(b) - testRunSize(a))[0];
  const testSelection = (heldout || []).filter(row => row.name === "zero-baseline" || (largestTestRun && row.name.startsWith(largestTestRun + "--")) || (row.historical && row.name.endsWith("--source-formula")));
  const testRows = population => testSelection.filter(row => row.population === population).map(row => ({
    ...row,
    targetName:row.name === "zero-baseline" ? "Zero change" : row.name.includes("fixed-hybrid") ? "Fixed temperature hybrid" : names[Object.keys(names).find(target => row.name.includes(target))] || row.name,
    control:row.historical ? "Historical / unequal cost" : row.name === "zero-baseline" ? "Baseline" : `${integer(testRunSize(largestTestRun) * 1000)} candidates / matched updates`,
  }));
  const gap = models.find(row=>row.seed === 20261009 && row.target === "state-boxcox" && row.architecture === "800x800x800x800" && row.updates === 10000 && row.trainingCount < 11000);
  const backbone = models.filter(row => row.seed === 20261009 && row.architecture === "800x800x800x800" && row.updates === 10000 && row.trainingCount > 40000 && row.trainingCount < 60000);
  const density = backbone.find(row => row.target === "state-boxcox");
  const densityTable = models.filter(row => row.seed === 20261009 && row.target === "state-boxcox" && row.precision === "float32" && row.architecture === "800x800x800x800" && row.updates === 10000).sort((a, b) => a.trainingCount - b.trainingCount);
  const seedRepeats = models.filter(row => row.seed === 20261010 && row.target === "state-boxcox");
  const repeatSmall = seedRepeats.find(row => row.trainingCount < 11000);
  const repeatLarge = seedRepeats.find(row => row.trainingCount > 40000 && row.trainingCount < 60000);
  const chartComparison = largestPrimaryComparison(models);
  const chartCount = chartComparison[0]?.trainingCount;
  const chartRows = chartComparison.map(row => ({...row, targetName:names[row.target]}));
  const selected = models.filter(row => row.precision === "float32" && row.trainingCount > 9000);
  const tableRows = selected.map(row => ({...row, targetName:names[row.target], recipe:`${row.architecture.startsWith("800") ? "Large GELU / L1" : "Small tanh / MSE"}; ${integer(row.updates)} updates`}));
  const sourcePreviews = {[paper]:{title:"Direct increment learning for combustion chemistry", source:"arXiv preprint", summary:"The study learns chemistry increments from augmented flame states. Its reported application includes a temperature-switched policy using transformed-state and direct-power models.", approvedForReport:true}};
  const prose = (id, title, queryId, rows, text) => visible(id) && <ReportSection id={id} title={title} queryId={queryId} sourceRows={rows} showHeading={false}><RichNarrative id={`${id}:body`} value={text} sourcePreviews={sourcePreviews} /></ReportSection>;
  const testTable = (population, title) => {
    const id = `flame-heldout-${population}`;
    const rows = testRows(population);
    return rows.length > 0 && visible(id) && <DataComponent id={id} queryId="heldout" kind="table" title={title} sourceRows={heldout.filter(row => row.population === population)} displayRows={rows}>
      <DataTable rows={rows} label={title} compactNumbers={false} columns={[
        {field:"control",label:"Control"}, {field:"targetName",label:"Target"}, {field:"samples",label:"Cells",renderCell:integer},
        {field:"budgetP99",label:"Budget p99 ↓",renderCell:budgetNumber}, {field:"heatRelativeRms",label:"Heat relative RMS ↓",renderCell:value=>value?.toFixed(3) ?? "—"},
        {field:"negativeEndpointFraction",label:"Negative species",renderCell:percent}, {field:"inverseCorrectionFraction",label:"Inverse corrections",renderCell:percent},
      ]} />
    </DataComponent>;
  };
  const binTable = (queryId, population, temperature) => {
    const allRows = snapshot.queries[queryId]?.rows;
    if (!allRows?.length) return null;
    const {rows, sourceRows} = policyBinTable(allRows, population, largestTestRun, temperature);
    const id = `flame-${queryId}`;
    const title = temperature ? "Temperature regions: balanced-sample budget p99" : "Increment magnitudes: uniform-sample absolute error p99";
    const format = value => value == null ? "Not observed" : temperature ? budgetNumber(value) : value.toExponential(2);
    return visible(id) && <DataComponent id={id} queryId={queryId} kind="table" title={title} sourceRows={sourceRows} displayRows={rows}>
      <DataTable rows={rows} label={title} searchable={false} compactNumbers={false} columns={[
        {field:"range",label:temperature ? "Initial temperature" : "|Reference increment|"},
        {field:"count",label:temperature ? "Cells" : "Species components",renderCell:integer},
        {field:"zero",label:"Zero",renderCell:format}, {field:"boxcox",label:"Conventional",renderCell:format},
        {field:"power",label:"Direct power",renderCell:format}, {field:"hybrid",label:"Fixed hybrid",renderCell:format},
      ]} />
    </DataComponent>;
  };
  return <article className="report-content" aria-label="Flame chemistry research review">
    <header className="report-hero">
      <h1 data-data-app-title contentEditable={canEdit && mode === "edit"} suppressContentEditableWarning onBlur={canEdit && mode === "edit" ? event => setAppTitle(event.currentTarget.textContent.trim() || appTitle) : undefined}>{appTitle}</h1>
      <RichNarrative id="flame:introduction" className="report-deck" value="We moved from a small numerical exercise to states from the NH₃/CH₄ flame application. The reference checks pass on selected states. The learned models still need work. No neural model has been installed in the CFD solver." />
    </header>
    {prose("flame-decision", "Current decision", "models", models,
      `## What this means now\n\nWe have a safe route from the existing flame case to checked chemistry labels, controlled model tests, and a copied CFD run. This is progress toward the application in the [Fuel study](${paper}). It is not a reproduction of the paper's accuracy.\n\nThe present evidence does not support replacing CVODE. The strict species errors remain large, and some predictions produce negative species fractions. Keep the deployment gate closed. The matched 10,000- and 50,000-state comparisons are complete and verified. All four targets improved on validation species p99 and heat-release RMS. This passed the planned gate for a 200,000-state density test; its label generation is running. The separate 2D test remains sealed.`)}
    {prose("flame-problem", "The problem we now learn", "models", models,
      "## 1. Learn the chemistry step that CFD actually calls\n\nEach input is a cell's temperature, pressure, and 59 species mass fractions. The target is the signed change in each species over one microsecond. Temperature and volume stay fixed during this local chemistry calculation. The CFD equations handle flow, transport, and energy outside it.\n\nThe earlier constant-pressure, adiabatic reactor audit asked a different question. It remains useful for studying numerical precision, but it cannot by itself represent this CFD application.\n\nWe used six time snapshots from the existing 1D flame. Four supply training states; two supply validation states. We split the snapshots before interpolation and perturbation. They still belong to one flame realization, so they are not six independent experiments.")}
    {prose("flame-input-precision", "Source input precision", "reference", reference,
      "The recovered 1D text samples have about six significant digits. We preserve the raw values and record the small mass-fraction normalization. The reference checks apply to these defined inputs. FP64 storage and tighter integration do not restore unknown digits from the original simulation.")}
    {profile && visible("flame-profile-chart") && <EvidenceChart id="flame-profile-chart" queryId="flame_profile" title="The copied NH₃/CH₄ flame: temperature before and after the short restart" rows={profile} sourceRows={profile}
      spec={{type:"line",x:"position_mm",y:"temperature_K",series:"time_ms",stackable:false,valueDecimals:2,xLabel:"Position (mm)",yLabel:"Temperature (K)"}} height={360} />}
    {prose("flame-labels", "Reference accuracy", "reference", reference,
      `## 2. Check the answers before judging the models\n\nCVODE generates the reference labels. Radau checks a selected subset by integrating the increments directly. This avoids relying only on subtraction of two endpoint values. Both methods still use Cantera's chemical rates; this is not an independent mechanism check.\n\nFor ${reference[0].states} augmented states, all ${integer(reference[0].speciesComponents)} species components passed the reference agreement check. The largest disagreement was ${reference[0].uncertaintyBudgetMax.toExponential(2)} times the chosen species error budget.\n\nThe separate relative-resolution screen passed ${integer(reference[0].resolvedNonzeroComponents)} of ${integer(reference[0].nonzeroReferenceComponents)} nonzero reference increments. There were ${integer(reference[0].unresolvedNonzeroComponents)} unresolved nonzero entries and ${integer(reference[0].zeroReferenceComponents)} zero reference entries. Report zeros separately: relative error at zero is undefined. A zero numerical reference alone is not proof of exact mathematical zero.\n\nThe screen requires an increment more than 100 times the larger of solver disagreement and endpoint spacing. It is roughly a 1% resolution screen, not evidence of twelve correct digits. These results apply to the checked subset, not every label.\n\nThe initial dataset kept 9,995 training states and 1,019 validation states. Five labels in each split failed the endpoint checks. Their records were retained and excluded, not replaced by zero.`)}
    {prose("flame-comparison", "Controlled comparison", "models", models,
      "## 3. Change how the target is represented\n\nWe compared four targets: transformed-state increments, direct signed-power increments, budget-linear increments, and scaled asinh increments. Within each comparison, the model, seed, data split, and update budget are the same.\n\nFirst, a small 128 × 128 × 128 tanh model used squared loss. FP32 and FP64 gave nearly the same result. More updates improved several results. Thus, arithmetic precision alone is not the main limit in this test.\n\nNext, we used the study's four 800-unit GELU layers and L1 loss. This changes both model capacity and training loss; it is not an isolated activation test. Our batch size and total training are still far below the original study. All normalization statistics use training data only.")}
    {visible("flame-results-table") && <DataComponent id="flame-results-table" queryId="models" kind="table" title="Completed FP32 comparisons on the same validation states" sourceRows={selected} displayRows={tableRows}>
      <DataTable rows={tableRows} label="Completed physical-space validation results" compactNumbers={false} columns={[
        {field:"recipe",label:"Training setup"}, {field:"seed",label:"Seed"}, {field:"trainingCount",label:"Training states",renderCell:integer}, {field:"targetName",label:"Target"}, {field:"budgetP99",label:"Budget error p99 ↓",renderCell:integer},
        {field:"trainingBudgetP99",label:"Training p99 ↓",renderCell:integer}, {field:"negativeEndpointFraction",label:"Negative species",renderCell:percent}, {field:"heatRelativeRms",label:"Heat RMS / reference RMS ↓",renderCell:value=>value.toFixed(3)},
      ]} />
    </DataComponent>}
    {gap && prose("flame-training-gap", "Training versus later flame states", "models", models,
      `### More samples are not the same as more flame coverage\n\nThe longer conventional model has heat-release relative RMS error ${gap.trainingHeatRelativeRms.toFixed(3)} on its training states and ${gap.heatRelativeRms.toFixed(3)} on the later validation states. Its species-budget p99 is ${integer(gap.trainingBudgetP99)} on training and ${integer(gap.budgetP99)} on validation.\n\nThis gap matters. The four training snapshots end at 1.5 ms; validation uses 2.0 and 2.5 ms. Adding perturbations around the same four snapshots does not add later source states. The 50,000-state run tests denser sampling of the existing coverage, not a broader flame history. We must distinguish these two changes before recommending a much larger dataset.`)}
    {prose("flame-metrics", "How to read the errors", "models", models,
      "## 4. Read the physical errors, not only the training loss\n\nFor each species component, divide the increment error by 10⁻¹² + 10⁻⁶ × |initial mass fraction|. A value of 1 meets this chosen budget. The p99 value is the error below which 99% of non-argon components fall. It is not a percentage. A value of 300,000 is still far outside the budget.\n\nThis is a strict research criterion. Passing it is not sufficient to prove stable CFD, and failing it does not by itself measure flame-speed error. Heat release is a second physical view. Its relative RMS error is 1 when the prediction is zero.\n\nBox–Cox inverse values outside their valid domain are mapped to zero and counted separately in the source evidence. Therefore, zero negative endpoints does not mean unconstrained predictions were all valid. Other models receive no hidden positivity or conservation repair.")}
    {density && gap && prose("flame-density-result", "What denser sampling changed", "models", models,
      `### Denser sampling helped under the same training budget\n\nFor the conventional target, increasing accepted training states from ${integer(gap.trainingCount)} to ${integer(density.trainingCount)} reduced validation species p99 from ${integer(gap.budgetP99)} to ${integer(density.budgetP99)}. Heat-release RMS error fell from ${percent(gap.heatRelativeRms)} to ${percent(density.heatRelativeRms)} of the reference RMS.\n\nAll four targets improved on both measures. Each fit still used 10,000 updates of 256 states, with the same model, seed, and validation states. The larger set therefore received fewer average presentations per state. This result supports one more measured density step, not an unlimited scale-up. It is one seed on correlated flame snapshots, not a statistical ranking.`)}
    {visible("flame-density-table") && <DataComponent id="flame-density-table" queryId="models" kind="table" title="Conventional target: completed data-size comparisons at fixed updates" sourceRows={densityTable} displayRows={densityTable}>
      <DataTable rows={densityTable} label="Primary-seed data-size comparison" searchable={false} compactNumbers={false} columns={[
        {field:"trainingCount",label:"Accepted training states",renderCell:integer},
        {field:"budgetP99",label:"Validation budget p99 ↓",renderCell:budgetNumber},
        {field:"heatRelativeRms",label:"Heat relative RMS ↓",renderCell:value=>value.toFixed(3)},
        {field:"selectedStep",label:"Selected update",renderCell:integer},
      ]} />
    </DataComponent>}
    {prose("flame-density-scope", "What remains fixed when data grows", "models", densityTable,
      "The table keeps all completed primary-seed conventional fits, not only the best result. Each fit uses the same 4×800 model and 10,000-update budget. The validation rule can select different saved updates. Input and target scales are fitted again on each training set. Thus, this compares the full training protocol at each size; it does not isolate data count with fixed normalization. A larger fit enters only after completion and verification.")}
    {repeatSmall && repeatLarge && prose("flame-seed-repeat", "Fixed second-seed check", "models", seedRepeats,
      `### Repeat the density question with another initialization\n\nThe conventional-only repeat uses fixed seed 20261010. The source data, model, 10,000 updates, batch size, and selection rule stay the same. At 10k candidates, validation species p99 was ${integer(repeatSmall.budgetP99)} and heat relative RMS was ${repeatSmall.heatRelativeRms.toFixed(3)}. At 50k candidates, these values were ${integer(repeatLarge.budgetP99)} and ${repeatLarge.heatRelativeRms.toFixed(3)}.\n\nBoth fits remain in the frozen test list. We did not choose the better seed. This checks one part of sensitivity to training randomness; it is not a repeated four-target ranking or a confidence interval.`)}
    {chartRows.length > 0 && visible("flame-heat-chart") && <EvidenceChart id="flame-heat-chart" queryId="models" title={`Heat-release error: ${integer(chartCount)} accepted states, 10,000 matched updates, primary seed`} rows={chartRows} sourceRows={chartComparison} spec={{type:"horizontalBar",x:"targetName",y:"heatRelativeRms",stackable:false,valueDecimals:3,xLabel:"Target",yLabel:"RMS error / reference RMS"}} height={340} />}
    {chartRows.length > 0 && visible("flame-species-validation") && <DataComponent id="flame-species-validation" queryId="models" kind="table" title={`Selected species: ${integer(chartCount)}-state primary-seed validation comparison`} sourceRows={chartComparison} displayRows={chartRows}>
      <DataTable rows={chartRows} label="Fuel, NO, and radical increment errors" columns={[{field:"targetName",label:"Target"},...speciesColumns]} />
    </DataComponent>}
    {prose("flame-species-limit", "What the species view means", "models", backbone,
      "NH₃ and CH₄ are the fuels in this case. NO and OH give separate views of a pollutant species and a radical. These four species were selected by chemical role, not by their test scores. Each value is the p99 of that species' increment error divided by its chosen state budget. It is not final NO emissions, a species concentration profile, or flame speed.")}
    {datasets && visible("flame-dataset-table") && <DataComponent id="flame-dataset-table" queryId="datasets" kind="table" title="Completed label generation" sourceRows={datasets} displayRows={datasets}>
      <DataTable rows={datasets} label="Dataset counts" columns={[{field:"candidates",label:"Training candidates",renderCell:integer},{field:"trainingAccepted",label:"Accepted training states",renderCell:integer},{field:"validationAccepted",label:"Accepted validation states",renderCell:integer},{field:"generationSeconds",label:"Generation time (seconds)",renderCell:integer}]} />
    </DataComponent>}
    {expanded && prose("flame-expanded-reference", "Larger-data checks", "expanded_reference", expanded,
      `### The larger data set passes the same subset check\n\nAll ${integer(expanded[0].components)} components in ${expanded[0].states} selected states passed the independent agreement check. The largest disagreement was ${expanded[0].uncertaintyBudgetMax.toExponential(2)} of the species budget. The original 10,000 training candidates are an exact prefix of the larger set. Validation inputs, accepted masks, and labels are identical. This makes the data-size comparison meaningful; it does not certify every label.`)}
    {historical && prose("flame-historical-context", "Existing trained models", "historical", historical,
      "### What the existing trained models tell us\n\nThe server already contained conventional and direct-power models trained with much more data and computation. We checked their identities and converted copies into plain numerical arrays. The original files and working environments were not changed.\n\nThe table below uses the original-style reconstruction formula. These models are not an equal-budget comparison with our short fits. Their old training data may overlap this validation domain. Treat them as a useful control, not new generalization evidence.\n\nThe fixed hybrid uses no change below 305 K, direct power from 305 to 1000 K, and the conventional model above 1000 K. This distinction matters: a model intended for a temperature region can look poor when used everywhere. Stable reconstruction alone made little change to the reported tail errors for these weights.")}
    {historical && visible("flame-historical-table") && <DataComponent id="flame-historical-table" queryId="historical" kind="table" title="Existing-weight controls: same 1D validation states, unequal training cost" sourceRows={historical} displayRows={historical.filter(row=>row.name.endsWith("source-formula"))}>
      <DataTable rows={historical.filter(row=>row.name.endsWith("source-formula"))} label="Historical model controls" columns={[{field:"name",label:"Control"},{field:"budgetP99",label:"Budget error p99 ↓",renderCell:integer},{field:"negativeEndpointFraction",label:"Negative species",renderCell:percent},{field:"heatRelativeRms",label:"Heat RMS / reference RMS ↓",renderCell:value=>value.toFixed(3)},{field:"inverseDomainViolationFraction",label:"Raw inverse violations",renderCell:value=>value == null ? "Not recorded" : percent(value)}]} />
    </DataComponent>}
    {historical && prose("flame-inverse-limits", "Inverse violations are not repairs", "historical", historical,
      "An invalid inverse-transform base and a performed correction are different things. The historical source formula can return a number from an invalid base without correcting it. Therefore, zero recorded corrections do not prove that all inverse operations were valid. Raw inverse violations are shown when recorded; a missing diagnostic is not zero.")}
    {filterAudit && prose("flame-filter-context", "Historical data filtering", "filter_audit", filterAudit,
      "### One difference from the old training data\n\nThe old source script keeps states whose formation-enthalpy change is at most 200 J/kg. This rejects sufficiently endothermic changes, not every negative heat-release value. Our current data do not apply this filter.\n\nThe rule would remove about 3.5–3.6% of accepted training states and 5.0% of validation states. Most rejected validation states are in the hottest bin. Removing these validation rows alone does not remove the large model errors. This does not tell us what retraining on filtered data would do. We kept the current comparison unchanged.")}
    {filterAudit && visible("flame-filter-table") && <DataComponent id="flame-filter-table" queryId="filter_audit" kind="table" title="Read-only effect of the historical filter" sourceRows={filterAudit} displayRows={filterAudit}>
      <DataTable rows={filterAudit} label="Historical-filter sensitivity" columns={[{field:"trainingCount",label:"Dataset training states",renderCell:integer},{field:"split",label:"Split"},{field:"states",label:"Accepted labels",renderCell:integer},{field:"rejected",label:"Would reject",renderCell:integer},{field:"rejectedFraction",label:"Fraction",renderCell:percent}]} />
    </DataComponent>}
    {heldout && prose("flame-heldout-context", "Reserved CFD snapshot", "heldout", heldout,
      "## The separate 2D snapshot: two different questions\n\nAll model identities and the 305/1000 K switching thresholds were fixed before this snapshot was opened. No model was trained or selected from these test scores. The uniform sample asks how the model performs on randomly selected cells. The temperature-balanced sample gives cold, preheat, reaction, and burnt regions a separate diagnostic view.\n\nDo not combine the two populations. Their cells can overlap, and the balanced sample is not a domain-average estimate. The main tables show the largest completed matched-data run, the zero-change baseline, and the preselected historical source-formula controls. This display rule was fixed before scoring; all frozen scores remain in the source data.\n\nOnly the new models were kept from this snapshot during this work. Prior exposure of the historical weights cannot be excluded. One offline snapshot does not prove stable CFD or accurate flame speed.")}
    {testSampling && visible("flame-heldout-sampling") && <DataComponent id="flame-heldout-sampling" queryId="heldout_sampling" kind="table" title="Test cells and accepted reference labels" sourceRows={testSampling} displayRows={testSampling}>
      <DataTable rows={testSampling} label="Separate test populations" columns={[{field:"population",label:"Population"},{field:"selected",label:"Selected cells",renderCell:integer},{field:"accepted",label:"Accepted labels",renderCell:integer},{field:"excluded",label:"Excluded labels",renderCell:integer}]} />
    </DataComponent>}
    {testReference && prose("flame-heldout-reference", "Independent test reference check", "heldout_reference", testReference,
      `### Check the test answers too\n\nThe independent test audit checked ${testReference[0].states} states and ${integer(testReference[0].speciesComponents)} species components. All passed the empirical budget-agreement check. Maximum disagreement was ${testReference[0].uncertaintyBudgetMax.toExponential(2)} times the budget.\n\nOf ${integer(testReference[0].nonzeroReferenceComponents)} nonzero numerical references, ${integer(testReference[0].resolvedNonzeroComponents)} passed the separate relative-resolution screen and ${integer(testReference[0].unresolvedNonzeroComponents)} did not. Another ${integer(testReference[0].zeroReferenceComponents)} references were zero. These are separate counts, not a claim that every tiny test increment has many correct digits.`)}
    {heldout && testTable("uniform", "Reserved 2D snapshot: uniform random cells")}
    {heldout && testTable("balanced", "Reserved 2D snapshot: temperature-balanced diagnostic")}
    {heldout && visible("flame-species-test") && <DataComponent id="flame-species-test" queryId="heldout" kind="table" title="Selected species: temperature-balanced 2D diagnostic" sourceRows={heldout.filter(row=>row.population === "balanced")} displayRows={testRows("balanced")}>
      <DataTable rows={testRows("balanced")} label="Selected species on the balanced test population" columns={[{field:"control",label:"Control"},{field:"targetName",label:"Target"},...speciesColumns]} />
    </DataComponent>}
    {heldout && prose("flame-bin-context", "Read the fixed policy by region", "heldout", heldout,
      "### Where does the temperature policy help?\n\nThe next tables retain the largest completed primary run's conventional, direct-power, and fixed-hybrid predictions, plus zero change. These columns and bin edges were fixed before scoring. The balanced temperature view gives each occupied region a diagnostic sample; it is not a domain average.\n\nThe magnitude view uses uniform random cells and counts species components, not cells. Its values are absolute increment errors, not budget or relative errors. The lowest bin includes numerical zeros. A small absolute error there does not establish many correct relative digits. Empty bins remain unobserved, not zero error. Other model scores and both populations remain in the source data.")}
    {heldout && binTable("heldout_temperature", "balanced", true)}
    {heldout && binTable("heldout_magnitude", "uniform", false)}
    {prose("flame-cfd", "Copied CFD baseline", "cfd", cfd,
      `## 5. Check the installed CFD solver without changing it\n\nA copied 500-cell case completed 100 steps, from 2.5 to 2.6 ms. Neural chemistry was disabled. The final maximum temperature was ${cfd[1].temperature_max_K.toFixed(2)} K. No final species component was negative. Maximum mass-fraction closure error was ${cfd[1].mass_closure_max.toExponential(2)}.\n\nAll ${cfd[1].originalFilesUnchanged} original files used by the copy retained their hashes. The installed image and shared environments were not changed. The copied case needed compatible energy-solver names and an inactive spray-cloud dictionary.\n\nThis proves that the existing runtime can execute the copied restart. It does not validate flame speed, mesh convergence, a long trajectory, or a learned chemistry model. The CFD runtime uses Cantera 2.6.0; the research labels use 3.2.0. That version difference remains explicit.`)}
    {tolerance && prose("flame-cfd-tolerance", "Numerical tolerance control", "cfd_tolerance", tolerance,
      `### Tightening chemistry tolerances changes this short run only slightly\n\nWe made a second case copy. It used the same mesh, initial fields, timestep, and installed solver. Neural chemistry stayed off. Only CVODE tolerances changed: relative/absolute values of 10⁻⁶/10⁻¹⁰ became 10⁻¹²/10⁻²¹.\n\nBoth copies completed ${tolerance[0].steps} steps. The largest final temperature difference was ${tolerance[0].temperatureMaxAbsK.toExponential(2)} K. The largest species mass-fraction difference was ${tolerance[0].speciesMaxAbs.toExponential(2)}. All ${tolerance[0].originalFilesUnchanged} original input files retained their hashes.\n\nThe final-state species-budget p99 was ${tolerance[0].finalStateBudgetP99.toFixed(2)}. This uses the tighter final mass fraction in the budget. Do not compare it directly with the one-step learned-increment scores above. This control shows tolerance sensitivity over this short restart. It does not prove an exact solution, mesh or timestep convergence, or neural-model accuracy.`)}
    {parity && prose("flame-runtime-parity", "Runtime version agreement", "runtime_parity", parity,
      `### The older runtime agrees on a checked subset\n\nThe unchanged CFD Cantera ${parity[0].runtimeCantera} environment was tested on ${parity[0].states} validation states. Its tight chemistry increments differed from the Cantera ${parity[0].researchCantera} labels by at most ${parity[0].maxDifferenceBudget.toExponential(2)} of the species error budget. This reduces a compatibility concern without upgrading the installed solver. It does not certify every state.`)}
    {prose("flame-next", "Next decision", "models", models,
      "## 6. Increase data only through a measured comparison\n\nThe next bounded run increases candidate training states from 50,000 to 200,000. It keeps the four source snapshots, validation states, model, and training budget fixed. Label generation uses separate one-hour segments and new output folders. Audit the completed labels before fitting. Do not start the final fit after 06:30 China time; leave time to verify and save the 08:00 checkpoint.\n\nAfter this pre-test decision and all included model identities are fixed, score the separate 2D CFD snapshot. Keep the random-cell sample separate from a temperature-balanced diagnostic sample. Also test the paper's fixed temperature-switch policy without fitting its thresholds to the test set.\n\nIf these tests still show large physical failures, repair the model or its constraints before any neural CFD run. More perturbations do not supply new flame conditions. The next coverage study must preserve a separate test set.\n\nReview and decisions remain in [Issue #3](https://github.com/xiao312/DFODE-kit/issues/3); implementation remains in [draft PR #4](https://github.com/xiao312/DFODE-kit/pull/4). This page presents the saved evidence, not a production release.")}
  </article>;
}
