// Default is a read-only freshness check. --write regenerates the review page.
import {readFileSync, writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {representations, recipes, recipeAliases} from "../benchmarks/flame_conditioning/report-content/method-catalogue.mjs";

const destination = new URL("../docs/research-method-catalogue.md", import.meta.url);
function diagram(steps) {
  return ["```mermaid", "flowchart TD",
    ...steps.map((step,i)=>`  n${i}["${step.replaceAll('"',"'")}"]`),
    ...steps.slice(1).map((_,i)=>`  n${i} --> n${i+1}`), "```"].join("\n");
}

export function renderCatalogue() {
  const lines = ["# Chemistry method catalogue", "",
    "> Generated from [the canonical registry](../benchmarks/flame_conditioning/report-content/method-catalogue.mjs). Do not edit this page by hand. Run `node scripts/build_research_method_catalogue.mjs --write` after changing the registry or renderer.", "",
    "This is the naming and diagram reference for the offline representation study. The [live report](https://xiao312.github.io/DFODE-kit/flame-conditioning/) contains measured results and the split/tolerance review guide. [Issue #3](https://github.com/xiao312/DFODE-kit/issues/3) records decisions. Old comments and artifact IDs remain historical records; use the mappings here to read them.", "",
    "The [accuracy protocol](agents/representation-accuracy-success-sources.md#required-dual-scaling-protocol-for-later-runs) distinguishes CVODE tolerance parameters, state-based error scaling, increment-based error scaling and custom magnitude-dependent budgets. Later runs must compare both error scales; the existing results are unchanged.", "",
    "## Three different concepts", "",
    "1. **Representation:** the coordinate system for a signed physical increment.",
    "2. **Method recipe:** input features, representation, approximator, loss and optimization.",
    "3. **Run:** recipe plus dataset, split, seed, precision and work budget.", "",
    "Changing a seed does not create a new method. Sharing an asinh transform does not make two recipes identical. The historical `arrhenius-*` IDs mean inverse-temperature and log-partial-pressure features, not an exact Arrhenius law. GBCT, full ISAT and spectrum-informed MSNN are not implemented by these recipes.", "",
    "## Data split sequence", "",
    diagram(["Split source cases or snapshots before augmentation", "Training: fit weights and learned scales", "Development: compare and revise recipes", "Freeze recipe and acceptance rule", "Independent-case test: evaluate once"]), "",
    "Current campaign: 10,000 selected training states per seed from 10,010 accepted rows; 1,023 development states. Four training and two development snapshots come from one flame realization. Development has been inspected repeatedly. Current independent-test performance is **not evaluated**, not zero. Older CFD test results concern older models. Local-table training scores are resubstitution scores, not leave-one-out validation.", "",
    "## Representation index", "",
    "Here d=delta Y, Y is the initial mass fraction, and z is the encoded target before recipe-specific scaling. Encoding uses reference labels during fitting. Inference predicts z from inputs and applies the inverse; it cannot inspect the unknown reference increment."];
  for (const row of representations) lines.push("", `<a id="representation-${row.id}"></a>`, `### ${row.name}`, "",
    `Representation ID: \`${row.id}\`.`, "", `\`${row.formula}\``, "", row.detail, "", diagram(row.flow));
  lines.push("", "## Method index", "", "Click a method name for its steps. Diagrams summarize the recipe, including fitting where labelled; they are not per-query cost diagrams.", "",
    "| Stable artifact ID | Canonical name | Family |", "| --- | --- | --- |");
  for (const row of recipes) lines.push(`| \`${row.id}\` | [${row.name}](#${row.id}) | ${row.family} |`);
  lines.push("", "Report aliases: " + Object.entries(recipeAliases).map(([alias,id])=>`\`${alias}\` → \`${id}\``).join(", ") + ".",
    "", "The zero-increment control always returns zero; it is not fitted.", "", "## Method steps");
  for (const row of recipes) lines.push("", `<a id="${row.id}"></a>`, `### ${row.name}`, "",
    `Artifact ID: \`${row.id}\`. Family: ${row.family}.`, "", row.detail, "", diagram(row.flow));
  lines.push("", "## Executable sources", "",
    "- [Original coordinates](../benchmarks/flame_conditioning/coordinates.py) and [learning settings](../benchmarks/offline_accuracy/learning.json).",
    "- [Refinement plan](../benchmarks/offline_accuracy/refinement/plan.py) and [coordinates](../benchmarks/offline_accuracy/refinement/coordinates.py).",
    "- [Adaptation plan](../benchmarks/offline_accuracy/improve/plan.py), [neural fitting](../benchmarks/offline_accuracy/improve/neural.py), [local fitting](../benchmarks/offline_accuracy/improve/local.py), and [input/head adaptation](../benchmarks/offline_accuracy/improve/arrhenius.py).",
    "- [Non-learned controls](../benchmarks/offline_accuracy/improve/physics_prior.py).", "",
    "Executable plans and saved run configurations own numerical parameters. This registry owns review names, diagrams and alias mappings. A display-name change never rewrites a run artifact.", "");
  return lines.join("\n");
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  if (process.argv.slice(2).some(arg=>arg !== "--write")) throw new Error("Use no arguments to check, or --write to regenerate");
  const rendered = renderCatalogue();
  if (process.argv.includes("--write")) writeFileSync(destination, rendered);
  else if (readFileSync(destination,"utf8").replaceAll("\r\n","\n") !== rendered) throw new Error("Catalogue is stale: run with --write");
  console.log(process.argv.includes("--write") ? "Catalogue generated." : "Catalogue matches registry.");
}
