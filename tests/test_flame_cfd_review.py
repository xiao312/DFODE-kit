import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.review_cfd import scalar_field, x_cell_centres


def test_reads_ascii_scalar_uniform_and_nonuniform(tmp_path):
    field = tmp_path / "T"
    header = "FoamFile {format ascii; class volScalarField;}\n"
    field.write_text(header + "internalField uniform 300;")
    np.testing.assert_array_equal(scalar_field(field, 3), [300, 300, 300])
    field.write_text(header + "internalField nonuniform List<scalar> 3 (1 2 3);")
    np.testing.assert_array_equal(scalar_field(field, 3), [1, 2, 3])
    with pytest.raises(ValueError):
        scalar_field(field, 4)


def test_nonuniform_geometry_is_not_replaced_by_uniform_spacing(tmp_path):
    case, original = tmp_path / "case", tmp_path / "original"
    (case / "constant/polyMesh").mkdir(parents=True)
    (original / "0").mkdir(parents=True)
    points = "\n".join(f"({x} {y} {z})" for x in (0, 1, 3) for y in (0, 1) for z in (0, 1))
    (case / "constant/polyMesh/points").write_text(points)
    geometry = original / "0/C"
    geometry.write_text("class volVectorField; internalField nonuniform List<vector> 2\n(\n(.5 .5 .5)\n(2 .5 .5)\n);\n")
    centres, path = x_cell_centres(case, original, 2)
    np.testing.assert_array_equal(centres, [.5, 2])
    assert path == geometry
    geometry.write_text("class volScalarField;")
    with pytest.raises(ValueError):
        x_cell_centres(case, original, 2)
