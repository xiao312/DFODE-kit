import json
from pathlib import Path
import time

import numpy as np
import pytest

pytest.importorskip("cantera")
torch = pytest.importorskip("torch")
from benchmarks.flame_conditioning.train import fit_variant, network, validate_config
from benchmarks.flame_conditioning.verify import load_predictor


def settings():
    return json.loads((Path(__file__).parents[1] / "benchmarks/flame_conditioning/learning.json").read_text())


def test_network_seed_and_dtype_match():
    model32 = network(5, 3, [8], 123, torch.float32)
    model64 = network(5, 3, [8], 123, torch.float64)
    for first, second in zip(model32.parameters(), model64.parameters(), strict=True):
        np.testing.assert_array_equal(first.detach().double().numpy(), second.detach().numpy())


def test_config_rejects_unknown_target_and_unbounded_job():
    config = settings()
    validate_config(config)
    config["targets"] = ["unknown"]
    with pytest.raises(ValueError):
        validate_config(config)
    config = settings()
    config["wall_seconds"] = 10000
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("target", ["state-boxcox", "signed-power", "budget-linear", "scaled-asinh"])
@pytest.mark.parametrize("activation,loss", [("tanh", "mse"), ("gelu", "l1")])
def test_tiny_training_saves_replayable_evidence(tmp_path, target, activation, loss):
    torch.set_num_threads(1)
    config = settings()
    config.update(updates=2, validation_every=1, hidden_widths=[8], batch_size=4, activation=activation, loss=loss)
    fractions = np.linspace(.1, .2, 12)
    states = np.column_stack([np.linspace(300, 1500, 12), np.full(12, 101325), fractions, .9-fractions, np.full(12, .1)])
    change = fractions * .01
    delta = np.column_stack([-change, change, np.zeros(12)])
    training = {"states": states, "delta": delta, "source_indices": np.arange(12)}
    validation = {key: value[:4] for key, value in training.items()}
    physics = {"species_names": ["A", "B", "AR"], "element_matrix": np.ones((1, 3)),
               "formation_enthalpies": np.array([0, -100000, 0]), "molecular_weights": np.array([2, 4, 40])}
    result = fit_variant(training, validation, physics, config, target, "float64", tmp_path / target, time.monotonic()+30)
    assert result["updates_completed"] == 2
    assert result["status"] == "complete"
    assert result["inactive_species"] == ["AR"]
    weights = torch.load(tmp_path / target / "weights.pt", weights_only=True)
    model = network(5, 3, [8], config["seed"], torch.float64, activation)
    model.load_state_dict(weights)
    assert (tmp_path / target / "validation-predictions.npz").is_file()
    predict, _ = load_predictor(tmp_path / target, config)
    recovered, mask = predict(validation["states"])
    saved = np.load(tmp_path / target / "validation-predictions.npz")
    np.testing.assert_array_equal(recovered, saved["prediction"])
    np.testing.assert_array_equal(mask, saved["correction"])
