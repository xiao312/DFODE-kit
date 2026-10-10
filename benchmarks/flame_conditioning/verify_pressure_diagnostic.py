"""Check saved pressure-only diagnostic identities, reference checks and scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cantera as ct
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.flame_conditioning.extract import sha256
from benchmarks.flame_conditioning.chemistry import element_matrix
from benchmarks.flame_conditioning.verify_physical import recompute, assert_scores


def verify(directory, test):
    summary = json.loads((directory / "summary.json").read_text())
    plan = summary["plan"]
    manifest = json.loads((test / "manifest.json").read_text())
    if summary["status"] != "complete" or sha256(test / "manifest.json") != plan["test_manifest_sha256"]:
        raise ValueError("Require complete diagnostic on the frozen test")
    for name, key in (("source-states.npz", "states_sha256"), ("labels.npz", "labels_sha256"),
                      ("mechanism.yaml", "mechanism_sha256")):
        if sha256(test / name) != manifest[key]:
            raise ValueError("Test artifact identity mismatch")
    if sha256(directory / "paired-reference.npz") != summary["paired_reference_sha256"]:
        raise ValueError("Paired reference identity mismatch")
    with np.load(directory / "paired-reference.npz", allow_pickle=False) as arrays:
        original, changed = arrays["original_states"], arrays["changed_states"]
        old_delta, new_delta = arrays["original_delta"], arrays["changed_delta"]
        indices, cells = arrays["source_rows"], arrays["cell_ids"]
    if len(indices) != 32 or len(set(indices.tolist())) != 32:
        raise ValueError("Require 32 distinct fixed audit states")
    np.testing.assert_array_equal(indices, plan["rows"])
    np.testing.assert_array_equal(cells, plan["cells"])
    np.testing.assert_array_equal(changed[:, 0], original[:, 0])
    np.testing.assert_array_equal(changed[:, 2:], original[:, 2:])
    np.testing.assert_array_equal(changed[:, 1], np.full(32, plan["pressure_Pa"]))
    with np.load(test / "source-states.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(original, arrays["states"][indices])
        np.testing.assert_array_equal(cells, arrays["sample_row"][indices])
    with np.load(test / "labels.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(old_delta, arrays["delta"][indices])
        assert arrays["accepted"][indices].all()
    gas = ct.Solution(str(test / "mechanism.yaml"))
    assert gas.species_names == manifest["species_names"]
    elements = element_matrix(gas)
    records = [json.loads(line) for line in (directory / "reference-records.jsonl").read_text().splitlines()]
    assert len(records) == 32
    errors = []
    for row, record in enumerate(records):
        assert record["row"] == indices[row] and record["cell"] == cells[row]
        state = record["state"]
        np.testing.assert_array_equal([state["T"], state["P"], *state["Y"]], changed[row])
        assert set(record["checks"]) == {"cvode", "step_limited", "direct", "direct_tight"}
        np.testing.assert_array_equal(new_delta[row], record["checks"]["cvode"]["delta"])
        row_errors = []
        for check in record["checks"].values():
            delta = np.asarray(check["delta"])
            assert delta.shape == new_delta[row].shape and np.isfinite(delta).all()
            assert (changed[row, 2:] + delta).min() >= -1e-21
            assert abs(delta.sum()) <= 1e-10 and np.max(np.abs(elements @ delta)) <= 1e-10
            assert abs(check["diagnostics"]["temperature_change_K"]) <= 1e-8
            assert abs(check["diagnostics"]["relative_density_change"]) <= 1e-10
            row_errors.append(float(np.max(np.abs(delta - new_delta[row]) / (1e-12 + 1e-6 * abs(changed[row, 2:])))))
        errors.append(max(row_errors))
        assert errors[-1] <= .01 and record["reference_pass"]
        np.testing.assert_allclose(errors[-1], record["uncertainty_budget_max"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(max(errors), summary["uncertainty_budget_max"], rtol=1e-12, atol=0)
    gas.TP = 298.15, ct.one_atm
    enthalpy = gas.partial_molar_enthalpies / gas.molecular_weights

    def heat(states, delta):
        values = []
        for state, increment in zip(states, delta, strict=True):
            gas.TPY = state[0], state[1], state[2:]
            values.append(-float(increment @ enthalpy) * gas.density / 1e-6)
        return np.array(values)

    rms = lambda values: np.linalg.norm(values) / np.sqrt(len(values))
    baseline = rms(heat(original, old_delta))
    response = rms(heat(changed, new_delta) - heat(original, old_delta)) / baseline
    np.testing.assert_allclose(response, summary["reference_heat_response_relative_rms"], rtol=2e-12)
    expected = {"state-boxcox", "signed-power", "budget-linear", "scaled-asinh", "fixed-hybrid", "zero-baseline"}
    assert len(summary["models"]) == 6 and {row["target"] for row in summary["models"]} == expected
    for model in summary["models"]:
        path = directory / f'{model["target"]}-paired-predictions.npz'
        assert sha256(path) == model["predictions_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            prediction, correction = arrays["prediction"], arrays["correction"]
        assert prediction.shape == (64, 59) and correction.shape == prediction.shape
        for name, states, delta, selection in (("original", original, old_delta, slice(0, 32)),
                                               ("training_mean_pressure", changed, new_delta, slice(32, 64))):
            scores = model["conditions"][name]
            assert_scores(recompute(states, prediction[selection], delta, test / "mechanism.yaml", 1e-6), scores)
            active = np.array([name != "AR" for name in gas.species_names])
            np.testing.assert_allclose(correction[selection][:, active].mean(), scores["inverse_domain_correction_fraction"])
        response = rms(heat(changed, prediction[32:]) - heat(original, prediction[:32])) / baseline
        np.testing.assert_allclose(response, model["prediction_heat_response_relative_rms"], rtol=2e-12)
    return {"status": "verified", "states": 32, "models": 6, "score_records": 12,
            "original_inputs_and_labels_unchanged": True, "only_pressure_changed": True,
            "reference_records_verified": 32, "uncertainty_budget_max": max(errors),
            "diagnostic_summary_sha256": sha256(directory / "summary.json"),
            "test_manifest_sha256": sha256(test / "manifest.json"),
            "scope": "Saved post-score subset diagnosis, not a new test or model repair."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Output must be new")
    result = verify(args.diagnostic, args.test)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
