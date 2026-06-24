import numpy as np
import torch

from dfode_kit.models.latent_baseline import ConservedLatentDeltaModel


def test_conserved_delta_model_outputs_atom_conserving_species_delta():
    completion = np.array([[-1.0], [-8.0], [9.0]], dtype=np.float64)
    mass_fraction_element_matrix = torch.tensor([[1.0, 0.0, 1.0 / 9.0], [0.0, 1.0, 1.0 / 18.0]], dtype=torch.float64)
    model = ConservedLatentDeltaModel(input_dim=5, completion_matrix=completion, latent_dim=2, hidden_dim=4)
    x = torch.randn(3, 5)
    current_y = torch.rand(3, 3, dtype=torch.float64)
    log_dt = torch.zeros(3, 1)

    out = model(x, log_dt, current_species=current_y)
    residual = out["delta_y"] @ mass_fraction_element_matrix.T

    assert out["next_state"].shape == (3, 5)
    assert out["delta_key"].shape == (3, 1)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)
