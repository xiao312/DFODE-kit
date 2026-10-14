# Verification Loop

The repository should provide a fast path for agents to validate changes before opening a PR.

## Required local loop
0. `make bootstrap-harness` — create local `uv`-managed `.venv` with lightweight verification deps
1. `make check` — fast syntax validation (`compileall`)
2. `make test` — lightweight pytest suite
3. `make verify` — all of the above

## Low-volume GitHub Actions policy

Routine pushes to research and feature branches do not launch the harness.
Run the relevant checks locally for each change. Run `make verify` before a
merge proposal. A saved successful check applies only to the commit it tested.

The harness runs when a PR is opened, reopened or marked ready for review,
on pushes to `main`, and on explicit workflow dispatch. PR updates
(`synchronize`) do not trigger it. New runs cancel an older harness run for
the same PR or branch. This is event reduction, not a daily quota.

Once the dispatch-enabled workflow is available on the default branch, a
maintainer can request a deliberate milestone check with:

```bash
gh workflow run harness.yml --ref <branch>
```

Until then, use local verification or the normal ready-for-review event.
Do not merge this research branch merely to install the dispatch button.
Do not toggle PR state just to trigger checks after each experiment.

The live research Pages site is published from `research-pages`. Batch site
updates at meaningful review milestones; do not push a site rebuild for each
research commit. This policy does not disable Pages or change its source.

GitHub documents the [workflow trigger activity types](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

## What belongs in lightweight tests
- Pure utility behavior
- File/schema/layout invariants
- Documentation or harness expectations
- Fast parser or serializer behavior

## What should be added later
- CLI smoke tests
- Training config schema validation tests
- Dataset contract validation tests
- Small synthetic end-to-end smoke runs

## Rule
If a rule matters for future agent-generated diffs, prefer encoding it as:
1. a test,
2. a script target,
3. CI,
4. documentation.
