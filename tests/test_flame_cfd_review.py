import numpy as np
import pytest

pytest.importorskip("cantera")
from benchmarks.flame_conditioning.review_cfd import scalar_field


def test_reads_ascii_scalar_uniform_and_nonuniform(tmp_path):
    field = tmp_path / "T"
    header = "FoamFile {format ascii; class volScalarField;}\n"
    field.write_text(header + "internalField uniform 300;")
    np.testing.assert_array_equal(scalar_field(field, 3), [300, 300, 300])
    field.write_text(header + "internalField nonuniform List<scalar> 3 (1 2 3);")
    np.testing.assert_array_equal(scalar_field(field, 3), [1, 2, 3])
    with pytest.raises(ValueError):
        scalar_field(field, 4)
