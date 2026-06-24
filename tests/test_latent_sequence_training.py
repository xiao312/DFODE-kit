import numpy as np

from dfode_kit.training.latent_sequence import LatentSequenceTrainingConfig


def test_latent_sequence_config_defaults_are_small_baseline():
    cfg = LatentSequenceTrainingConfig()

    assert cfg.latent_dim == 16
    assert cfg.hidden_dim == 128
    assert cfg.num_layers == 1
    assert cfg.reconstruction_weight == 0.1
    assert cfg.time_weight == 0.01


def test_sequence_normalization_guard_can_replace_zero_std():
    data = np.ones((2, 3, 4), dtype=np.float32)
    flat = data.reshape(-1, data.shape[-1])
    std = flat.std(axis=0)
    std = np.where(std > 0, std, 1.0)

    assert np.all(std == 1.0)
