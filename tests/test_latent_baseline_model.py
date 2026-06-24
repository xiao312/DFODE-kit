import torch

from dfode_kit.models.latent_baseline import AutoencoderLatentGRU


def test_autoencoder_latent_gru_shapes():
    model = AutoencoderLatentGRU(input_dim=11, latent_dim=4, hidden_dim=8, condition_dim=3)
    x = torch.randn(2, 5, 11)

    decoded_next, z_next = model(x)
    time_event = model.predict_time_event(z_next, condition=torch.randn(2, 4, 3))

    assert decoded_next.shape == (2, 4, 11)
    assert z_next.shape == (2, 4, 4)
    assert time_event["log_dt"].shape == (2, 4, 1)
    assert time_event["equilibrium_logit"].shape == (2, 4, 1)
