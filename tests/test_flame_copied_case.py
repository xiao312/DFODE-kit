import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.copy_case import adapt_energy_solvers, chemistry_dictionary, control_dictionary


def test_control_has_literal_bounds_and_no_executable_includes():
    text = control_dictionary(100)
    assert "endTime 0.00260000;" in text
    assert "functions {}" in text
    assert "#" not in text
    assert "writePrecision 17;" in text
    assert "adjustTimeStep off;" in text
    with pytest.raises(ValueError):
        control_dictionary(101)


def test_adaptation_is_explicit_and_fails_on_unknown_source():
    original = '"(U|ha|k|epsilon)" {solver PBiCGStab;} "(U|ha|k|epsilon)Final" {$U;}'
    modified = adapt_energy_solvers(original)
    assert modified.count("U|h|hs|ha|k|epsilon") == 2
    assert "h|hs" not in original
    with pytest.raises(ValueError):
        adapt_energy_solvers("unknown")


def test_chemistry_preserves_original_tolerances_and_disables_nn():
    text = chemistry_dictionary("/case/mechanism.yaml")
    assert "relTol 1e-6;" in text and "absTol 1e-10;" in text
    assert "torch false;" in text and "GPU false;" in text
    assert "active false;" in text
    assert "inertSpecie AR;" in text
