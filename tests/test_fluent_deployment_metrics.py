import numpy as np
import pytest

from dfode_kit.evaluation.fluent_deployment import (
    conditioned_source_summaries,
    density_increment_nmae,
    endpoint_from_delta,
    limiter_conditioned_summary,
    species_source_from_increment,
    species_source_summary,
    summarize_fluent_deployment,
)


def test_density_increment_and_source_conversion_use_physical_increments():
    rho_now = np.array([1.0, 2.0])
    rho_true = np.array([1.1, 1.8])
    rho_pred = np.array([1.05, 1.9])
    assert density_increment_nmae(rho_now, rho_true, rho_pred) == pytest.approx(
        0.5
    )

    current = np.array([[0.8, 0.2], [0.5, 0.5]])
    target = np.array([[0.7, 0.3], [0.55, 0.45]])
    source = species_source_from_increment(
        current, target, density=np.array([2.0, 1.0]), dt=np.array([0.5, 0.25])
    )
    np.testing.assert_allclose(source, [[-0.4, 0.4], [0.2, -0.2]])


def test_species_source_metrics_cover_reactive_and_conditioned_cells():
    true = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]])
    pred = np.array([[0.1, -0.1], [0.8, -0.8], [2.2, -2.2]])
    summary = species_source_summary(
        true, pred, ["FUEL", "PRODUCT"], reactive_threshold=0.5
    )
    assert summary["reactive_cell_count"] == 2
    assert summary["per_species"][0]["species"] == "FUEL"

    conditioned = conditioned_source_summaries(
        true,
        pred,
        bin_edges={
            "dt": [0.0, 2.0, 4.0],
            "temperature": [250.0, 1000.0, 2000.0],
            "mixture_fraction": [0.0, 0.5, 1.0],
            "progress_variable": [0.0, 0.25, 1.0],
        },
        dt=[1.0, 2.0, 3.0],
        temperature=[300.0, 900.0, 1500.0],
        mixture_fraction=[0.1, 0.4, 0.8],
        progress_variable=[0.0, 0.2, 0.9],
    )
    assert set(conditioned) == {
        "dt",
        "temperature",
        "mixture_fraction",
        "progress_variable",
    }
    assert conditioned["dt"][0]["sample_count"] == 1
    assert conditioned["dt"][1]["sample_count"] == 2

    limiter = limiter_conditioned_summary(true, pred, [False, True, False])
    assert limiter["activation_rate"] == pytest.approx(1.0 / 3.0)
    assert limiter["active"]["sample_count"] == 1
    assert limiter["inactive"]["sample_count"] == 2


def test_combined_summary_reports_deployment_quantities():
    source_true = np.array([[1.0, -1.0], [0.5, -0.5]])
    source_pred = np.array([[0.9, -0.9], [0.6, -0.6]])
    summary = summarize_fluent_deployment(
        density_current=[1.0, 1.0],
        density_true=[1.1, 0.9],
        density_pred=[1.08, 0.92],
        mixture_molecular_weight_true=[28.0, 30.0],
        mixture_molecular_weight_pred=[28.1, 29.8],
        enthalpy_true=[1.0e6, 1.2e6],
        enthalpy_pred=[1.01e6, 1.19e6],
        heat_release_true=[2.0e8, -1.0e8],
        heat_release_pred=[1.9e8, -1.1e8],
        species_source_true=source_true,
        species_source_pred=source_pred,
        species_names=["A", "B"],
        limiter_active=[False, True],
    )
    assert summary["density_increment"]["nmae"] == pytest.approx(0.2)
    assert summary["heat_release"]["cosine"] > 0.99
    assert summary["mixture_molecular_weight"]["mae"] == pytest.approx(0.15)
    assert summary["species_source_by_limiter"]["activation_count"] == 1


def test_summary_separates_raw_corrected_and_deployed_stages():
    current = np.array(
        [[1000.0, 101325.0, 0.8, 0.2], [1200.0, 101325.0, 0.6, 0.4]]
    )
    raw = endpoint_from_delta(
        current,
        delta_temperature=[10.0, 20.0],
        delta_mass_fractions=[[-0.81, 0.81], [-0.02, 0.02]],
    )
    corrected = raw.copy()
    corrected[0, 2:] = [0.0, 1.0]
    target = current.copy()
    target[:, 0] += [12.0, 18.0]
    target[:, 2:] += [[-0.1, 0.1], [-0.01, 0.01]]
    source_true = np.array([[-1.0, 1.0], [-0.5, 0.5]])
    source_pred = np.array([[-0.9, 0.9], [-0.4, 0.4]])

    summary = summarize_fluent_deployment(
        density_current=[1.0, 1.0],
        density_true=[0.99, 0.98],
        density_pred=[0.991, 0.979],
        mixture_molecular_weight_true=[28.0, 29.0],
        mixture_molecular_weight_pred=[28.1, 28.9],
        enthalpy_true=[1.0, 2.0],
        enthalpy_pred=[1.1, 1.9],
        heat_release_true=[3.0, 4.0],
        heat_release_pred=[2.9, 4.1],
        species_source_true=source_true,
        species_source_pred=source_pred,
        species_names=["A", "B"],
        target_endpoint=target,
        raw_network_endpoint=raw,
        post_positivity_endpoint=corrected,
    )
    stages = summary["prediction_stages"]
    assert set(stages) >= {
        "raw_network_endpoint",
        "post_positivity_endpoint",
        "deployed_source",
    }
    assert stages["raw_network_endpoint"]["negative_cell_rate"] == 0.5
    assert stages["post_positivity_endpoint"]["negative_cell_rate"] == 0.0
