from copy import deepcopy

import pytest

from benchmarks.offline_accuracy.paper_baseline.matched_work.plan import configuration, extended_configuration, update_policy
from benchmarks.offline_accuracy.paper_baseline.matched_work.review import pair_rows, validate_result, POLICIES


def checked_fixture(recipe="fuel-state", count=50000, extended=False):
    config = (extended_configuration if extended else configuration)(recipe, 20261011)
    updates = config["updates"]
    schedule, previous = [], None
    for completed in range(updates):
        rate, reset = update_policy(config, completed)
        if rate != previous:
            schedule.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
            previous = rate
    result = dict(status="complete", artifacts={"environment.json":"env"}, config=config,
        training_count=count, development_count=1023, independent_test_count=0,
        updates_completed=updates, row_presentations=updates*10000, effective_batch_size=10000,
        parameter_count=2018458, completed_pool_passes=updates*10000//count, schedule=schedule,
        history=[dict(updates=n) for n in range(1000, updates+1, 1000)])
    check = dict(status="verified", result_sha256="result", exact_model_replay=True,
        independent_paired_counts=True, physical_checks=True, frozen_50k_preprocessing=True)
    return result, check


@pytest.mark.parametrize("recipe", ["fuel-state", "fuel-power"])
@pytest.mark.parametrize("count", [50000, 200000])
def test_review_requires_exact_work_and_verification(recipe, count):
    result, check = checked_fixture(recipe, count)
    args = ("result", "env", recipe, 20261011, count)
    validate_result(result, check, *args)
    for key, value in (("row_presentations", 60000001), ("independent_test_count", 1),
                       ("effective_batch_size", 20000), ("development_count", 1024)):
        changed = deepcopy(result)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_result(changed, check, *args)
    for key in ("physical_checks", "frozen_50k_preprocessing", "exact_model_replay"):
        with pytest.raises(ValueError):
            validate_result(result, dict(check, **{key:False}), *args)
    with pytest.raises(ValueError):
        validate_result(result, check, "wrong hash", *args[1:])


def test_pairs_keep_both_seeds_policies_and_percentage_point_units():
    rows = [dict(recipe=recipe, seed=seed, policy=policy, trainingCount=count,
                 developmentRate=.4 if count == 50000 else .3)
            for recipe in ("fuel-state-matched-work", "fuel-power-matched-work")
            for seed in (20261011, 20261012) for policy in POLICIES for count in (50000, 200000)]
    pairs = pair_rows(rows)
    assert len(pairs) == 8
    assert all(r["changePoints"] == pytest.approx(-10.) for r in pairs)
    with pytest.raises(ValueError):
        pair_rows(rows[:-1])
    with pytest.raises(ValueError):
        pair_rows(rows+[rows[0]])


@pytest.mark.parametrize("recipe", ["fuel-state", "fuel-power"])
def test_extended_review_rejects_partial_work_or_default_protocol(recipe):
    result, check = checked_fixture(recipe, 200000, extended=True)
    args = ("result", "env", recipe, 20261011, 200000)
    validate_result(result, check, *args, extended=True)
    with pytest.raises(ValueError):
        validate_result(result, check, *args)
    with pytest.raises(ValueError):
        validate_result(dict(result, row_presentations=60000000), check, *args, extended=True)


def test_budget_pairs_preserve_repeats_and_units():
    from benchmarks.offline_accuracy.paper_baseline.matched_work.budget_review import pairs_from_rows
    rows = [dict(recipe=recipe, seed=seed, policy=policy, updates=updates,
                 developmentRate=.4 if updates == 6000 else .6, fitSeconds=10 if updates == 6000 else 30)
            for recipe in ("fuel-state-budget", "fuel-power-budget") for seed in (20261011,20261012)
            for policy in POLICIES for updates in (6000,18000)]
    pairs = pairs_from_rows(rows)
    assert len(pairs) == 8
    assert all(row["changePoints"] == pytest.approx(20) and row["wallRatio"] == 3 for row in pairs)
    with pytest.raises(ValueError):
        pairs_from_rows(rows[:-1])
    with pytest.raises(ValueError):
        pairs_from_rows(rows+[rows[0]])


def test_budget_collector_reads_full_baseline_contract_before_campaign(tmp_path, monkeypatch):
    from benchmarks.offline_accuracy.paper_baseline.matched_work import budget_review
    monkeypatch.setattr(budget_review, 'collect_baseline', lambda path: ([], [], [], [], ('data','audit','original')))
    with pytest.raises(FileNotFoundError, match='verification.json'):
        budget_review.collect(tmp_path, tmp_path / 'baseline')
