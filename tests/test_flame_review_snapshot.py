import pytest

from benchmarks.flame_conditioning.review_snapshot import reference_row


def test_reference_report_separates_zero_and_unresolved_nonzero():
    audit = {"components_checked": 4, "budget_fit_fraction": 1.0, "records": [
        {"reference_delta": [0.0, 1e-12, -1e-20, 1e-32],
         "relative_fit": [False, True, True, False], "uncertainty_budget_max": 1e-5}]}
    row = reference_row(audit)
    assert row["zeroReferenceComponents"] == 1
    assert row["nonzeroReferenceComponents"] == 3
    assert row["resolvedNonzeroComponents"] == 2
    assert row["unresolvedNonzeroComponents"] == 1
    assert row["relativeFitFraction"] == .5
    audit["components_checked"] = 5
    with pytest.raises(ValueError):
        reference_row(audit)
