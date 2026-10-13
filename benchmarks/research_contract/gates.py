"""Conjunctive development gates; no gate can offset a failure in another."""
import math


def comparison(candidate, comparator, policy):
    failures, missing = [], []

    def minimum(name, value, limit):
        if value is None or limit is None or not math.isfinite(value) or not math.isfinite(limit):
            missing.append(name)
        elif value < limit:
            failures.append(name)

    def maximum(name, value, limit):
        minimum(name, None if value is None else -value, None if limit is None else -limit)

    cg, bg = candidate["groups"], comparator["groups"]
    # The candidate score must have been formed against this exact comparator.
    if any(cg[k]["base_pass_count"] != bg[k]["pass_count"]
           or cg[k]["components"] != bg[k]["components"] for k in cg):
        raise ValueError("Comparator counts/masks differ from candidate scoring")
    minimum("overall_gain", cg["all"]["acceptance_gain"], policy["minimum_gain_fraction"])
    minimum("nontrivial_gain", cg["nontrivial"]["acceptance_gain"], 0.)
    for name in ("weak_nontrivial", "active_increment"):
        if cg[name]["components"] < policy["minimum_group_components_for_pilot_gate"]:
            missing.append(name + "_support")
        else:
            minimum(name + "_gain", cg[name]["acceptance_gain"], -policy["primary_group_max_drop_fraction"])
    if set(candidate["per_species"]) != set(comparator["per_species"]):
        raise ValueError("Species group registry differs")
    for name, group in candidate["per_species"].items():
        if group["components"] >= policy["minimum_group_components_for_pilot_gate"]:
            minimum("species_gain:" + name, group["acceptance_gain"], -policy["supported_species_max_drop_fraction"])
    maximum("damage", cg["all"]["damage_rate_among_base_successes"], policy["maximum_damage_rate"])
    for name in ("all", "weak_nontrivial"):
        p99 = bg[name]["normalized_error"]["p99"]
        maximum(name + "_p99", cg[name]["normalized_error"]["p99"],
                None if p99 is None else policy["maximum_p99_ratio"] * p99)
    maximum("wrong_sign", candidate["nontrivial_sign"]["wrong_sign_count"],
            comparator["nontrivial_sign"]["wrong_sign_count"])
    events = {(v["epsilon"], v["eta"]): v for v in comparator["small_target_events"]}
    if len(events) != len(candidate["small_target_events"]):
        raise ValueError("Event grid differs")
    for value in candidate["small_target_events"]:
        key = value["epsilon"], value["eta"]
        if key not in events or value["components"] != events[key]["components"]:
            raise ValueError("Event populations differ")
        if not value["components"]:
            missing.append(f"event_support:{key}")
        else:
            maximum(f"event:{key}", value["events"], events[key]["events"])
    if not candidate["returned_prediction_finite"]:
        failures.append("nonfinite_prediction")
    if candidate["argon_nonzero_or_nonfinite_returned_count"]:
        failures.append("argon_invariant")
    if "endpoint_diagnostics" not in candidate or "endpoint_diagnostics" not in comparator:
        missing.append("endpoint_diagnostics")
    else:
        maximum("negative_endpoints", candidate["endpoint_diagnostics"]["negative_components"],
                comparator["endpoint_diagnostics"]["negative_components"])
        maximum("nonfinite_endpoints", candidate["endpoint_diagnostics"]["nonfinite_components"], 0)
    return dict(failures=failures, missing=missing, passed=not failures and not missing)


def decision(comparisons, expected_keys, latency_ratio, policy):
    if set(comparisons) != set(expected_keys):
        return dict(candidate_status="INCONCLUSIVE", reason="Missing seed/population/comparator evidence")
    failures = {k: v["failures"] for k, v in comparisons.items() if v["failures"]}
    missing = {k: v["missing"] for k, v in comparisons.items() if v["missing"]}
    if latency_ratio is None or not math.isfinite(latency_ratio) or latency_ratio <= 0:
        missing["inference"] = ["Missing matched end-to-end latency"]
    elif latency_ratio > policy["maximum_inference_latency_ratio"]:
        failures["inference"] = ["Inference budget exceeded"]
    status = "ADMISSIBLE_DEVELOPMENT"
    if failures:
        status = "REJECTED" if any("overall_gain" in v for v in failures.values()) else "TRADEOFF_ONLY"
    elif missing:
        status = "INCONCLUSIVE"
    return dict(candidate_status=status, failures=failures, missing=missing,
                production_promotion=False, independently_confirmed=False)
