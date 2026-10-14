"""Independently reconcile a reference-policy amendment before model scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.extract import sha256, source_revision


def inspect_checks(initial, saved, record, elements):
    """Independent calculations from raw arrays, not the recovery decision helper."""
    if "error" in record:
        return False, None
    if set(record["checks"]) != {"fresh", "tighter_absolute", "direct", "direct_tight"}:
        raise ValueError("Missing recovery solve")
    differences, admissible = [], True
    for check in record["checks"].values():
        delta = np.asarray(check["delta"])
        if delta.shape != saved.shape or not np.isfinite(delta).all():
            return False, None
        endpoint_values = initial + delta
        diagnostics = check["diagnostics"]
        np.testing.assert_allclose(endpoint_values.min(), diagnostics["minimum_mass_fraction"], rtol=1e-12, atol=0)
        admissible = admissible and (
            endpoint_values.min() >= -1e-21 and abs(delta.sum()) <= 1e-10
            and np.max(np.abs(elements @ delta)) <= 1e-10
            and abs(diagnostics["temperature_change_K"]) <= 1e-8
            and abs(diagnostics["relative_density_change"]) <= 1e-10)
        differences.append(float(np.max(np.abs(delta - saved) / (1e-12 + 1e-6 * np.abs(initial)))))
    error = max(differences)
    return bool(admissible and error <= .01), error if admissible else None


def verify(original, recovered):
    old = json.loads((original / "manifest.json").read_text())
    new = json.loads((recovered / "manifest.json").read_text())
    summary = json.loads((recovered / "recovery-summary.json").read_text())
    plan = new["reference_recovery"]
    if new["status"] not in ("complete", "complete_with_exclusions") or summary["status"] != "complete":
        raise ValueError("Recovery is incomplete")
    if (plan["original_test_manifest_sha256"] != sha256(original / "manifest.json")
            or plan["original_labels_sha256"] != sha256(original / "labels.npz")
            or summary["test_manifest_sha256"] != sha256(recovered / "manifest.json")
            or plan["negative_endpoint_floor"] != 1e-21 or plan["maximum_disagreement_budget"] != .01):
        raise ValueError("Recovery policy or source binding mismatch")
    for name, key in (("source-states.npz", "states_sha256"), ("mechanism.yaml", "mechanism_sha256"),
                      ("frozen-plan.json", "frozen_plan_sha256")):
        if not (sha256(original / name) == sha256(recovered / name) == old[key] == new[key]):
            raise ValueError("A frozen source or model-plan byte changed")
    if sha256(original / "label-records.jsonl") != sha256(recovered / "label-records.jsonl"):
        raise ValueError("Original diagnostic records changed")
    for name, key in (("labels.npz", "labels_sha256"), ("reference-quality.npz", "reference_quality_sha256"),
                      ("recovery-records.jsonl", "recovery_records_sha256")):
        if sha256(recovered / name) != new[key]:
            raise ValueError("Recovered artifact hash mismatch")
    with np.load(original / "labels.npz", allow_pickle=False) as arrays:
        delta, strict = arrays["delta"], arrays["accepted"]
    with np.load(recovered / "labels.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["delta"], delta)
        accepted = arrays["accepted"]
    with np.load(recovered / "reference-quality.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["strict_accepted"], strict)
        flags, uncertainty = arrays["recovered"], arrays["uncertainty_budget"]
    if np.any(strict & ~accepted) or np.any(flags & strict):
        raise ValueError("An old accepted label changed classification")
    np.testing.assert_array_equal(accepted, strict | flags)
    with np.load(original / "source-states.npz", allow_pickle=False) as arrays:
        states, cells = arrays["states"], arrays["sample_row"]
    gas = ct.Solution(str(original / "mechanism.yaml"))
    if gas.species_names != old["species_names"] or old["species_names"] != new["species_names"]:
        raise ValueError("Species order mismatch")
    elements = element_matrix(gas)
    records = [json.loads(line) for line in (recovered / "recovery-records.jsonl").read_text().splitlines()]
    original_records = [json.loads(line) for line in (original / "label-records.jsonl").read_text().splitlines()]
    failed = np.flatnonzero(~strict)
    if [row["row"] for row in records] != failed.tolist():
        raise ValueError("Missing, duplicated, or reordered recovery records")
    extrema = []
    for index, record in zip(failed, records, strict=True):
        if record["cell"] != cells[index]:
            raise ValueError("Recovery cell identity changed")
        initial = states[index, 2:]
        minimum = (initial + delta[index]).min()
        original_diagnostics = original_records[index]["diagnostics"]
        old_admissible = (minimum < 0 and minimum >= -1e-21 and abs(delta[index].sum()) <= 1e-10
                          and np.max(np.abs(elements @ delta[index])) <= 1e-10
                          and abs(original_diagnostics["temperature_change_K"]) <= 1e-8
                          and abs(original_diagnostics["relative_density_change"]) <= 1e-10)
        passed, error = inspect_checks(initial, delta[index], record, elements)
        if bool(accepted[index]) != bool(passed and old_admissible) or bool(flags[index]) != record["recovered"]:
            raise ValueError("Reference eligibility differs from independent calculation")
        if error is not None:
            np.testing.assert_allclose(uncertainty[index], error, rtol=1e-12, atol=0)
        extrema.append(float(minimum))
    if new["labels_accepted"] != int(accepted.sum()) or summary["recovered"] != int(flags.sum()):
        raise ValueError("Recovery summary count mismatch")
    return {"status": "verified", "source": source_revision(), "raw_signed_labels_unchanged": True,
            "source_cells_and_frozen_plan_unchanged": True, "original_acceptance_mask_preserved": True,
            "selected": len(states), "strict_accepted": int(strict.sum()), "recovered": int(flags.sum()),
            "accepted": int(accepted.sum()), "still_excluded": int((~accepted).sum()),
            "independently_checked_rejections": len(records), "most_negative_original_endpoint": min(extrema),
            "uncertainty_budget_max": float(np.nanmax(uncertainty)), "negative_endpoint_floor": 1e-21,
            "original_test_manifest_sha256": sha256(original / "manifest.json"),
            "test_manifest_sha256": sha256(recovered / "manifest.json"),
            "recovery_summary_sha256": sha256(recovered / "recovery-summary.json"),
            "scope": "Post-reference, pre-score eligibility amendment; raw signs retained, no exact positivity claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("recovered", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Verification output must be new")
    result = verify(args.original, args.recovered)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
