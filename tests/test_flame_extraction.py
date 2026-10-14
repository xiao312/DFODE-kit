from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.extract import normalize_sampled_states, read_snapshot


def write_pair(root, second_coordinates=(0, 1)):
    for suffix, name, rows in (
        ("A", "T_p_H2", [[0, 300, 101325, .25], [1, 1500, 101325, .1]]),
        ("B", "O2", [[second_coordinates[0], .75], [second_coordinates[1], .9]]),
    ):
        destination = Path(root) / "postProcessing" / f"lineSample{suffix}" / "0.001"
        destination.mkdir(parents=True)
        np.savetxt(destination / f"line{suffix}_{name}.xy", rows)


def test_reads_matching_columns_and_coordinates(tmp_path):
    write_pair(tmp_path)
    raw, corrected, coordinates, correction, files = read_snapshot(tmp_path, "0.001", ["H2", "O2"])
    assert raw.shape == (2, 4)
    np.testing.assert_array_equal(raw, corrected)
    assert coordinates.tolist() == [0, 1]
    assert len(files) == 2 and all(len(record["sha256"]) == 64 for record in files)
    assert np.max(correction["normalization_max_absolute_change"]) == 0


def test_rejects_reordered_species(tmp_path):
    write_pair(tmp_path)
    with pytest.raises(ValueError, match="species order"):
        read_snapshot(tmp_path, "0.001", ["O2", "H2"])


def test_rejects_unpaired_coordinates(tmp_path):
    write_pair(tmp_path, (1, 0))
    with pytest.raises(ValueError, match="coordinates"):
        read_snapshot(tmp_path, "0.001", ["H2", "O2"])


def test_records_normalization_and_preserves_source():
    source = np.array([[300, 101325, .2, .800001]])
    original = source.copy()
    corrected, report = normalize_sampled_states(source)
    np.testing.assert_array_equal(source, original)
    assert corrected[0, 2:].sum() == pytest.approx(1)
    assert report["raw_mass_sum_error"][0] == pytest.approx(1e-6)
    assert report["normalization_max_absolute_change"][0] > 0


@pytest.mark.parametrize("values", [[300, 101325, -.000001, 1.000001],
                                  [300, 101325, .2, .9], [0, 101325, .2, .8]])
def test_rejects_invalid_source(values):
    with pytest.raises(ValueError):
        normalize_sampled_states([values])
