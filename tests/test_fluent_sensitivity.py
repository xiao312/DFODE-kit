import h5py
import numpy as np
import pytest

from dfode_kit.data.fluent_sensitivity import (
    FLUENT_DI_JVP_SCHEMA_VERSION,
    FLUENT_NATIVE_DI_BACKEND,
    FluentDIJVPBatch,
    finite_difference_jvp,
    load_fluent_di_jvp_dataset,
    write_fluent_di_jvp_dataset,
)


def _batch(provenance=None):
    x = np.array(
        [
            [1000.0, 101325.0, 0.8, 0.2],
            [1200.0, 101325.0, 0.6, 0.4],
        ]
    )
    direction = np.array(
        [
            [10.0, 0.0, -0.1, 0.1],
            [-5.0, 0.0, 0.2, -0.2],
        ]
    )
    epsilon = np.array([1.0e-3, 2.0e-3])
    x_perturbed = x + epsilon[:, None] * direction
    phi = x + np.array([[1.0, 0.0, -0.01, 0.01], [2.0, 0.0, -0.02, 0.02]])
    phi_perturbed = phi + epsilon[:, None] * (2.0 * direction)
    return FluentDIJVPBatch(
        x=x,
        x_perturbed=x_perturbed,
        phi_di_x=phi,
        phi_di_x_perturbed=phi_perturbed,
        epsilon=epsilon,
        direction=direction,
        dt=np.array([1.0e-6, 1.0e-6]),
        species_names=["CH4", "N2"],
        provenance=provenance
        or {
            "label_backend": FLUENT_NATIVE_DI_BACKEND,
            "fluent_version": "2026R1",
            "case_id": "sandia-d-gri30-geko-urans-v1",
            "exact_dt_seconds": 1.0e-6,
        },
        sample_metadata={
            "cell_id": np.array([10, 20]),
            "direction_kind": np.array(["outer-iteration", "outer-iteration"]),
        },
    )


def test_finite_difference_jvp_and_hdf5_round_trip(tmp_path):
    batch = _batch()
    expected = 2.0 * batch.direction
    assert finite_difference_jvp(
        batch.phi_di_x, batch.phi_di_x_perturbed, batch.epsilon
    ) == pytest.approx(expected)

    path = tmp_path / "fluent-di-jvp.h5"
    report = write_fluent_di_jvp_dataset(path, batch)
    loaded = load_fluent_di_jvp_dataset(path)
    assert report["schema_version"] == FLUENT_DI_JVP_SCHEMA_VERSION
    assert loaded.jvp_target == pytest.approx(expected)
    assert loaded.sample_metadata["cell_id"].tolist() == [10, 20]
    assert loaded.sample_metadata["direction_kind"].tolist() == [
        "outer-iteration",
        "outer-iteration",
    ]
    assert loaded.provenance["case_id"] == "sandia-d-gri30-geko-urans-v1"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["label_backend"] == FLUENT_NATIVE_DI_BACKEND
        assert handle["pairs/direction"].dtype == np.dtype("float64")
        assert handle["pairs/epsilon"].dtype == np.dtype("float64")


@pytest.mark.parametrize("token", ["Cantera", "CVODE", "SUNDIALS"])
def test_rejects_non_fluent_label_provenance(token):
    provenance = {
        "label_backend": FLUENT_NATIVE_DI_BACKEND,
        "case_id": "sandia-d",
        "upstream_labeler": token,
        "exact_dt_seconds": 1.0e-6,
    }
    with pytest.raises(ValueError, match="cannot use"):
        _batch(provenance)


def test_rejects_inconsistent_perturbation_relation():
    batch = _batch()
    batch.x_perturbed[0, 0] += 1.0
    with pytest.raises(ValueError, match=r"epsilon \* direction"):
        batch.validate()


def test_requires_exact_dt_in_native_di_provenance():
    provenance = {
        "label_backend": FLUENT_NATIVE_DI_BACKEND,
        "fluent_version": "2026R1",
        "case_id": "sandia-d",
    }
    with pytest.raises(ValueError, match="exact_dt_seconds"):
        _batch(provenance)


def test_rejects_pair_dt_that_differs_from_provenance():
    batch = _batch()
    batch.dt[1] = 2.0e-6
    with pytest.raises(ValueError, match="exactly match"):
        batch.validate()
