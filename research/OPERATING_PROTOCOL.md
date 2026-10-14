# Operating protocol for a persistent research worker

## Owner amendment (2026-10-13)

The owner removed the proposed configuration, fit-count and time caps below.
The current `campaign.json` is authoritative for execution resources. Each
comparison still declares finite optimizer work before it starts. Use one idle
GPU; preserve all other users' work. Stop only this campaign's own work on resource
contention, then wait or report. API spending and confirmation remain disabled.
This amendment supersedes the starter kit's finite-epoch cap language, not the
scientific contract, evaluator isolation, review gates or experiment provenance.

## Design

Use a bounded, restartable research process rather than an indefinitely growing conversation. The system has three logical roles: planner/implementer, numerical evaluator, and reviewer. One coding-agent runtime can perform planning and implementation. The numerical evaluator must be a pinned deterministic program in a protected environment. A separate reviewer context checks scientific interpretation and code quality; a second LLM is not the authority on floating-point pass counts.

The supervisor owns job admission, budgets, leases, process handles, experiment state and promotion. The agent proposes the next action; tools produce actual artifacts. No statement of success is accepted without the corresponding result files and independent scoring.

## Startup

Read the existing repository instructions, CHARTER, current verified status, the active experiment manifest, evidence ledger and recent scoped Git history. Resolve active run IDs against actual scheduler/process state before starting anything. Verify the scorer/dataset/incumbent identities and the minimal baseline replay. A restart must not duplicate a still-running GPU experiment.

If a later repository update supersedes the attached starting evidence, incorporate it with citations and preserve the older record. Do not execute an obsolete experiment just because this package names it as the first proposal.

## One research cycle

1. Choose an unresolved question from the evidence ledger. Prefer a test whose possible outcomes distinguish competing hypotheses. A candidate recipe must name its predicted metric effects, not merely say 'try attention' or 'improve loss'.
2. Write an experiment manifest before model fitting: hypothesis, prior evidence, single changed factor or declared factorial design, controls, data exposure, all seeds, budgets, expected distinguishing outcomes, unchanged gates and code paths.
3. Check that the experiment is inside the approved action/resource envelope. The supervisor grants a run lease only after identity and budget checks. Lack of an idle approved GPU means WAITING_RESOURCE, not CPU fallback, job killing, or deployment failure.
4. Use an isolated worktree and new artifact directory. Implement the candidate and tests. Run algebraic/unit/replay checks first. Establish that dependencies and runtime match or explicitly classify an environment change.
5. Execute the prespecified experiment. Record actual optimizer updates, presentations, preprocessing/setup time, CPU/GPU resource use, total inference cost, precision flags, logs, model hashes and failures. Never choose a better seed or endpoint post hoc.
6. The protected evaluator loads the candidate in a sandbox with only permitted state inputs. Candidate code cannot mount evaluation labels. Recount physical metrics from saved predictions in a separate process. Include the fixed anchor and current incumbent.
7. The reviewer labels the result: VALID_NEGATIVE, VALID_POSITIVE, TRADEOFF_ONLY, INCONCLUSIVE, or INVALID. Record what is demonstrated, what remains hypothetical, and which future observation would change the interpretation.
8. Archive the trial, update the append-only ledger, write a restartable status note and commit scoped code. A valid scientific failure advances knowledge but does not promote its model. Promote only by supervisor action after objective gates and required independent checks.
9. Continue to the next justified experiment while approved budget remains. Do not ask the user for permission after each valid negative result. If no supported next test remains, complete the epoch with a decision report rather than inventing an unbounded sweep.

## Separate states

Experiment state: PROPOSED -> APPROVED -> IMPLEMENTED -> PRECHECKED -> QUEUED -> RUNNING -> SCORED -> REVIEWED -> ARCHIVED.

Scientific conclusion: SUPPORTED / UNSUPPORTED_IN_TESTED_SCOPE / INCONCLUSIVE / INVALID.

Candidate status: INCUMBENT / ADMISSIBLE_DEVELOPMENT / TRADEOFF_ONLY / REJECTED / PENDING_CONFIRMATION / CONFIRMED.

Never use one `success` boolean for all three meanings. Do not retroactively declare failed historical candidate trials successful because their code executed correctly.

## Bounded autonomy

A proposed initial research epoch has a maximum of 12 distinct candidate configurations and 24 primary fits (two seeds each), one concurrent GPU fit, 1,200 s fit-wall and 1,800 s total-task ceiling per fit, and 28,800 allocated GPU-seconds. Set an additional owner-approved API spending limit before running the agent. These are resource ceilings, not estimates of how long work will take.

Prespecify one paired screening budget per scientific comparison. Matched updates/presentations isolate recipe changes; separately report wall time. If a new method needs different work to test its plausible capability, create a budget-comparison experiment with matching controls rather than mislabeling unmatched work. An architecture search under fixed hardware wall time answers a cost-constrained question, not an architecture-only causal question.

Allow one exact retry for a documented transient infrastructure failure, with both attempts recorded and charged. A scientific negative result is not a retryable infrastructure event. A changed optimizer, seed, loss or checkpoint policy creates a new trial identity. Large empirical improvements must be replayed and confirmed, not automatically accepted.

After two properly controlled negative experiments in the same hypothesis family without new distinguishing evidence, mark that line blocked for the current epoch. It can be revisited with a materially changed prediction or control, not just a different name. This rule limits repeated blind sweeps; it does not establish scientific impossibility.

## What may be autonomous

Inside the approved domain and finite budget: read source and official literature, propose experiments, write scoped candidate code, change representations/features/architecture/optimization in declared comparisons, run tests, train on the approved pool, make bounded reference audits using existing methods, inspect diagnostics, retain negative evidence and choose the next hypothesis. Enforce existing resource and cluster rules, including Slurm where applicable.

Changes to the scientific contract, reference labels/version, role of a split, total compute allocation, new mechanism/domain, confirmation access, production paths, shared environments, remote permissions, publishing or merge-to-production require the owner's separate authorization. Research-scoped local commits are different from public pushes or issue posts; authorize those explicitly.

Protect the scorer and reference data with file/account/container permissions, not just prompts. Use allowlisted network access for literature and package provenance. Never disable all permissions to make the loop run. Candidate processes must not inherit credentials. They must not inspect other users' jobs or data, terminate other workloads, or acquire access by modifying network/security configuration.

## Recovery and durable state

Persist run ID, immutable manifest hash, worktree commit, scheduler job ID/PID, last verified checkpoint, artifact paths, budget spent/remaining, current hypothesis and next allowed action outside the agent's context window. Keep raw logs; the short status note is an index, not the only record.

On interruption, query the actual job. If it finished, collect and verify it. If it is still running, attach to its status rather than launch another. If partial outputs are invalid, mark them partial and rerun only under the retry policy. Unknown is not passed or failed scientifically.

## Grading the agent itself

Measure valid completed experiments, reproducible result fraction, baseline/denominator mistakes, unsupported numerical claims, violations of allowed scope, duplicate runs after resume, wasted compute on repeated failed families, and independently verified model improvements per resource budget. Do not reward number of commits, experiments, explanatory tokens or leaderboard updates alone.

Inject a small set of harness tests: lower training loss but worse acceptance; hidden label access; a changed reference denominator; an empty rare-event baseline; a GPU job that remains alive after a context reset; mixed-runtime timing; and a known valid negative experiment. The correct agent preserves evidence, avoids false promotion and continues or pauses for the right reason.

## First experiment family

The inspected evidence currently favors a frozen-hidden-feature comparison, not another loss or damping sweep. Use one predeclared 800-dimensional post-final-GELU base feature and compare (A) 119 inputs, widths [256,256,256], 177210 parameters; (B) 919 inputs, widths [256,256,256], 382010; (C) 119 inputs, widths [800,256,256], 381754. Keep the continued base, 10k data, Smooth L1 coordinate objective, decoder and paired budget fixed. B/C are near parameter-matched, not mathematically identical. Exact parameter initialization hashes cannot match different shapes; exact initial PHYSICAL predictions can match by zero output initialization. Preserve the old A evidence or rerun if unchanged-runtime parity cannot be established.

If B wins transfer while C does not, retain feature reuse as a supported direction. If both win similarly, capacity remains a plausible explanation. If only training improves, investigate transfer/regularization. If neither wins, do not silently sweep every hidden layer; use the result to choose one new discriminating hypothesis. The mission remains broader than this first family.
