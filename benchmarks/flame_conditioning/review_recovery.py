"""Sanitized review row for the explicit reference-policy amendment."""


def recovery_row(verified, heldout):
    if (verified["status"] != "verified" or not verified["raw_signed_labels_unchanged"]
            or not verified["source_cells_and_frozen_plan_unchanged"]
            or not verified["original_acceptance_mask_preserved"]
            or verified["test_manifest_sha256"] != heldout["test_manifest_sha256"]):
        raise ValueError("Reference amendment is unverified or differs from the scored test")
    if (verified["strict_accepted"] + verified["recovered"] != verified["accepted"]
            or verified["accepted"] + verified["still_excluded"] != verified["selected"]):
        raise ValueError("Reference amendment counts are inconsistent")
    fields = ("selected", "strict_accepted", "recovered", "accepted", "still_excluded",
              "independently_checked_rejections", "most_negative_original_endpoint",
              "uncertainty_budget_max", "negative_endpoint_floor")
    return {key: verified[key] for key in fields}
