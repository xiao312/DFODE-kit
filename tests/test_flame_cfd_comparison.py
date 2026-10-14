import json

import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.compare_cfd import check_inputs, difference_metrics
from benchmarks.flame_conditioning.copy_case import chemistry_dictionary
from benchmarks.flame_conditioning.extract import sha256


def test_controls_allow_only_declared_chemistry_difference(tmp_path):
    cases = [tmp_path / name for name in ("study", "tight")]
    for case, preset in zip(cases, ("study", "tight")):
        (case / "constant").mkdir(parents=True)
        (case / "initial").write_text("same initial states")
        chemistry = case / "constant/CanteraTorchProperties"
        chemistry.write_text(chemistry_dictionary(case / "mechanism.yaml", preset))
        plan = {"original_sha256": {"initial": "same"}, "mechanism_sha256": "same",
                "steps": 100, "start_time_s": .0025, "interval_s": 1e-6,
                "prepared_sha256": {name: sha256(case / name) for name in
                                    ("initial", "constant/CanteraTorchProperties")}}
        (case / "preparation.json").write_text(json.dumps(plan))
    assert len(check_inputs(*cases)) == 2
    (cases[1] / "initial").write_text("changed")
    with pytest.raises(ValueError, match="Prepared CFD input changed"):
        check_inputs(*cases)
    (cases[1] / "initial").write_text("same initial states")
    chemistry = cases[1] / "constant/CanteraTorchProperties"
    chemistry.write_text(chemistry.read_text().replace("torch false;", "torch true;"))
    with pytest.raises(ValueError, match="CVODE-only preset"):
        check_inputs(*cases)


def test_difference_metrics_use_physical_fields_and_exclude_argon_from_budget():
    tight = np.array([[300., 1e5, .5, .5]])
    study = np.array([[301., 1e5 + 2, .500001, .6]])
    result = difference_metrics(study, tight, ["H2", "AR"])
    assert result["temperature_max_abs_K"] == 1
    assert result["pressure_max_abs_Pa"] == 2
    assert result["non_argon_species_budget_p99"] == pytest.approx(1e-6 / (1e-12 + 5e-7))
    assert result["per_species_max_abs"]["AR"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="finite"):
        difference_metrics(study * np.nan, tight, ["H2", "AR"])
