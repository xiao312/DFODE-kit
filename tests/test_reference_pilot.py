import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("cantera")
pytest.importorskip("scipy")
MODULE = Path(__file__).parents[1] / "benchmarks/precision_conditioning/reference"


def load(name):
    spec = importlib.util.spec_from_file_location(name, MODULE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chemistry = load("chemistry")
analysis = load("analysis")
pilot = load("pilot")
CONFIG = json.loads((MODULE / "pilot.json").read_text())


def test_config_and_cantera_mechanism_metadata():
    pilot.validate(CONFIG)
    expected = [(10, 29), (53, 325)]
    for mechanism, counts in zip(CONFIG["mechanisms"], expected):
        metadata = chemistry.mechanism_metadata(mechanism)
        assert (metadata["species_count"], metadata["reaction_count"]) == counts
        assert len(metadata["sha256"]) == 64


def test_direct_increment_agrees_on_short_h2_interval():
    mechanism = CONFIG["mechanisms"][0]
    state = chemistry.initial_state(mechanism, 900, CONFIG)
    interval = 1e-8
    gas = chemistry.gas_for(mechanism)
    scale = chemistry.budgets(state, CONFIG)
    rate = chemistry.direct_rhs(gas, state, scale)(0, np.zeros(len(scale))) * scale
    independent = chemistry.direct_increment(mechanism, state, interval, CONFIG, 1e-11, 1e-7)
    endpoint = chemistry.endpoint(mechanism, state, interval, 1e-12, 1e-21)
    direct = np.asarray(independent["delta"])
    assert np.any(direct[1:] != 0)
    assert np.max(np.abs(direct - endpoint["delta"]) / scale) < .01
    # Tiny-time direct integral should also agree with the initial RHS to first order.
    assert np.max(np.abs(direct - interval * rate) / scale) < .01
    assert abs(independent["conservation"]["mass_delta_sum"]) < 1e-16


def test_assessment_does_not_call_small_or_missing_reference_resolved():
    config = CONFIG
    record = {"state": {"T": 900., "Y": [.1, 0.]}, "solutions": {}}
    assert analysis.assess(record, config) is None
    for key in ("absolute18", "absolute21", "step_limited", "radau_check", "radau_reference"):
        record["solutions"][key] = {"delta": [0., 1e-32, 0.]}
    record["solutions"]["absolute21"]["delta"][1] = 0
    assessment = analysis.assess(record, config)
    assert assessment["budget_fit"][1]
    assert not assessment["relative_fit"][1]
    assert not assessment["relative_fit"][2]
    assert assessment["endpoint_zero_direct_nonzero"][1]


def test_quantization_reports_normalization_separately():
    state = {"T": 900.123456, "P": 101325., "Y": [.1, .2, .7]}
    quantized, effects = chemistry.quantized_state(state)
    assert abs(sum(quantized["Y"]) - 1) < 1e-15
    assert effects["raw_Y_error_max"] > 0
    assert effects["normalization_change_max"] > 0


def test_summary_separates_input_effects_from_reference_error():
    conservation = {"mass_delta_sum": 0., "element_delta_max": 0.,
                    "enthalpy_relative_drift": 0., "minimum_mass_fraction": 0.}
    names = ("absolute18", "absolute21", "step_limited", "radau_check", "radau_reference", "fp32_input")
    record = {"id": "one", "mechanism": "test", "interval_s": 1e-6,
              "state": {"T": 900., "Y": [0., .1]}, "failures": {},
              "solutions": {name: {"delta": [0., 1e-8, 0.], "seconds": 1.,
                                   "conservation": conservation} for name in names}}
    record["solutions"]["absolute21"]["delta"] = [0., 1e-8 + 1e-15, 0.]
    record["solutions"]["fp32_input"]["delta"] = [0., 1e-8 + 3e-15, 0.]
    summary, components, _ = analysis.summarize([record], CONFIG, [{"id": "test", "species": ["A", "B"]}])
    assert summary["assessed_species_components"] == 2
    assert summary["budget_fit_count"] == 2
    assert summary["relative_fit_count"] == 1
    assert summary["reference_zero_estimate_count"] == 1
    quantized = summary["solvers"]["fp32_input"]
    assert quantized["input_quantization_budget_max_per_interval"]["max"] == pytest.approx(.002)
    assert quantized["species_budget_error_max_per_interval"]["max"] == pytest.approx(.003)
    assert components[0]["relative_fit"]


def test_report_escapes_table_values_and_refuses_incomplete_run(tmp_path):
    report = load("report")
    assert "&lt;script&gt;" in report.table(["Title"], [["<script>"]])
    (tmp_path / "summary.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="complete pilot"):
        report.build(tmp_path)
