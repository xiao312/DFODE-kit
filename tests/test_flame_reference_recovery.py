import numpy as np
import pytest

from benchmarks.flame_conditioning.recover_test_reference import constraints_pass, recovery_decision
from benchmarks.flame_conditioning.verify_reference_recovery import inspect_checks
from benchmarks.flame_conditioning.review_recovery import recovery_row


def diagnostics(minimum=-1e-36):
    return {"minimum_mass_fraction": minimum, "mass_delta_sum": 0., "element_delta_max": 0.,
            "temperature_change_K": 0., "relative_density_change": 0.}


def checks(delta):
    return {name: {"delta": delta.copy(), "diagnostics": diagnostics()} for name in
            ("fresh", "tighter_absolute", "direct", "direct_tight")}


def test_sign_noise_is_checked_without_clipping_any_input_or_label():
    initial, delta = np.array([1., 0.]), np.array([0., -1e-36])
    saved = delta.copy()
    assert not constraints_pass(diagnostics(), 0)  # Original strict rejection.
    passed, uncertainty = recovery_decision(initial, delta, diagnostics(), checks(delta))
    assert passed and uncertainty == 0
    np.testing.assert_array_equal(delta, saved)
    assert delta[1] < 0


def test_recovery_rejects_material_negativity_disagreement_or_missing_checks():
    initial, delta = np.array([1., 0.]), np.array([0., -1e-36])
    assert not recovery_decision(initial, delta, diagnostics(-1e-20), checks(delta))[0]
    changed = checks(delta)
    changed["direct_tight"]["delta"][1] += 2e-14
    assert not recovery_decision(initial, delta, diagnostics(), changed)[0]
    changed = checks(delta)
    changed["fresh"]["diagnostics"]["relative_density_change"] = 1e-5
    assert not recovery_decision(initial, delta, diagnostics(), changed)[0]
    with pytest.raises(ValueError, match="Every independent"):
        recovery_decision(initial, delta, diagnostics(), {"fresh": changed["fresh"]})


def test_independent_verifier_recalculates_endpoint_sign_and_disagreement():
    initial, delta = np.array([1., 0.]), np.array([0., -1e-36])
    record = {"checks": checks(delta)}
    passed, error = inspect_checks(initial, delta, record, np.array([[1., 1.]]))
    assert passed and error == 0
    record["checks"]["fresh"]["diagnostics"]["minimum_mass_fraction"] = 0
    with pytest.raises(AssertionError):
        inspect_checks(initial, delta, record, np.array([[1., 1.]]))


def test_report_amendment_requires_exact_test_binding_and_preservation():
    verified = {"status": "verified", "raw_signed_labels_unchanged": True,
                "source_cells_and_frozen_plan_unchanged": True, "original_acceptance_mask_preserved": True,
                "test_manifest_sha256": "test", "selected": 3, "strict_accepted": 1,
                "recovered": 2, "accepted": 3, "still_excluded": 0, "independently_checked_rejections": 2,
                "most_negative_original_endpoint": -1e-36, "uncertainty_budget_max": 1e-7,
                "negative_endpoint_floor": 1e-21}
    assert recovery_row(verified, {"test_manifest_sha256": "test"})["recovered"] == 2
    with pytest.raises(ValueError, match="differs from the scored test"):
        recovery_row(verified, {"test_manifest_sha256": "other"})
    with pytest.raises(ValueError, match="unverified"):
        recovery_row({**verified, "raw_signed_labels_unchanged": False}, {"test_manifest_sha256": "test"})
