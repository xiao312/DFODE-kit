// Bind review-guide provenance without changing evidence, identity or timestamps.
// Default validates only. --apply writes the generated snapshot after validation.
import {readFileSync, writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {recipeNames} from "../benchmarks/flame_conditioning/report-content/method-catalogue.mjs";

export function annotateReview(snapshot) {
  const updated = structuredClone(snapshot);
  for (const queryId of ["improve_models","refinement_models"]) {
    const query = updated.queries[queryId];
    if (!query?.source || !query.rows.length) throw new Error(`Missing reviewed query: ${queryId}`);
    for (const row of query.rows) {
      if (!recipeNames[row.name]) throw new Error(`Unregistered method: ${row.name}`);
    }
    const label = "Training, development and independent test";
    query.source.metricDefinitions = query.source.metricDefinitions.filter(row=>row.label !== label);
    query.source.metricDefinitions.push({label,
      componentIds:["review-split-guide","review-split-scores","review-tolerance-guide"],
      definition:"Saved final-checkpoint acceptance: abs(prediction-reference)<=1e-15+0.1*abs(reference), all 58 non-AR species for complete states. Train: 10,000 selected rows per seed, 580,000 components. Development: 1,023 rows, 59,334 components. Independent test has not been run for these models and is displayed as null, never zero. Both seeds retained. Local training scores include queried points in their own tables. Only 16 development states have independently audited references; full-population scores are nominal."});
  }
  return updated;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [path,...flags] = process.argv.slice(2);
  if (!path || flags.some(flag=>flag !== "--apply")) throw new Error("Usage: node scripts/prepare_research_review.mjs <snapshot.json> [--apply]");
  const updated = annotateReview(JSON.parse(readFileSync(path,"utf8")));
  if (flags.includes("--apply")) writeFileSync(path, JSON.stringify(updated,null,2)+"\n");
  console.log(flags.includes("--apply") ? "Review metadata saved; evidence rows unchanged." : "Review metadata validation passed; no file changed.");
}
