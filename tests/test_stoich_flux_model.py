import numpy as np
import torch

from dfode_kit.models.latent_baseline import StoichiometricFluxModel


def test_stoich_flux_model_outputs_mass_conserving_delta_for_balanced_reaction():
    # H2 + 0.5 O2 -> H2O, converted to mass-fraction delta columns W*S.
    stoich_mass = np.array([[-2.0], [-16.0], [18.0]], dtype=np.float64)
    model = StoichiometricFluxModel(input_dim=5, stoichiometric_mass_matrix=stoich_mass, latent_dim=2, hidden_dim=4)
    x = torch.randn(3, 5)
    current_y = torch.rand(3, 3, dtype=torch.float64)
    log_dt = torch.zeros(3, 1)

    out = model(x, log_dt, current_species=current_y)

    assert out["next_state"].shape == (3, 5)
    assert out["reaction_extent"].shape == (3, 1)
    assert torch.allclose(out["delta_y"].sum(dim=-1), torch.zeros(3, dtype=torch.float64), atol=1e-10)
