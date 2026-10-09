from copy import deepcopy

import pytest

from benchmarks.offline_accuracy.paper_baseline.matched_work.plan import configuration, update_policy
from benchmarks.offline_accuracy.paper_baseline.matched_work.review import pair_rows, validate_result, POLICIES


def checked_fixture(recipe="fuel-state", count=50000):
    config = configuration(recipe, 20261011)
    schedule, previous = [], None
    for completed in range(6000):
        rate, reset = update_policy(config, completed)
        if rate != previous:
            schedule.append(dict(first_update=completed+1, learning_rate=rate, reset_adam=reset))
            previous = rate
    result = dict(status="complete", artifacts={"environment.json":"env"}, config=config,
        training_count=count, development_count=1023, independent_test_count=0,
        updates_completed=6000, row_presentations=60000000, effective_batch_size=10000,
        parameter_count=2018458, completed_pool_passes=60000000//count, schedule=schedule,
        history=[dict(updates=n) for n in range(1000, 6001, 1000)])
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
