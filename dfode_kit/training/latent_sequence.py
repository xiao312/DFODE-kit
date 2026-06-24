from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch

from dfode_kit.models.latent_baseline import AutoencoderLatentGRU


@dataclass(frozen=True)
class LatentSequenceTrainingConfig:
    latent_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 1
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    reconstruction_weight: float = 0.1
    time_weight: float = 0.01


def load_sequence_arrays(source_path: str, *, dtype=np.float32) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with h5py.File(source_path, "r") as h5:
        if "trajectories" in h5:
            sequences = [group["states"][:] for _, group in sorted(h5["trajectories"].items())]
            times = [group["times"][:] for _, group in sorted(h5["trajectories"].items())]
        else:
            sequences = [dataset[:] for _, dataset in sorted(h5["sequences"].items())]
            dt = float(h5.attrs.get("dt", 1.0))
            times = [np.arange(sequence.shape[0], dtype=np.float64) * dt for sequence in sequences]
        species_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in h5["species_names"][:]
        ]

    if not sequences:
        raise ValueError(f"No sequences found in {source_path}")

    shapes = {sequence.shape for sequence in sequences}
    if len(shapes) != 1:
        raise ValueError(f"All sequences must have the same shape; got {sorted(shapes)}")

    time_shapes = {time.shape for time in times}
    if len(time_shapes) != 1:
        raise ValueError(f"All time arrays must have the same shape; got {sorted(time_shapes)}")

    return (
        np.stack(sequences, axis=0).astype(dtype),
        np.stack(times, axis=0).astype(dtype),
        species_names,
    )


def load_sequence_tensor(source_path: str) -> tuple[np.ndarray, list[str]]:
    sequences, _times, species_names = load_sequence_arrays(source_path)
    return sequences, species_names


def train_latent_sequence_model(
    source_path: str,
    output_path: str,
    *,
    config: LatentSequenceTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, float]:
    cfg = config or LatentSequenceTrainingConfig()
    raw_sequences, times, species_names = load_sequence_arrays(source_path)

    flat = raw_sequences.reshape(-1, raw_sequences.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    sequences = (raw_sequences - mean) / std

    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    tensor = torch.tensor(sequences, dtype=torch.float32)
    time_deltas = np.diff(times, axis=1)
    log_time_deltas = np.log(np.maximum(time_deltas, 1e-300)).astype(np.float32)
    time_tensor = torch.tensor(log_time_deltas[..., None], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor, time_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = AutoencoderLatentGRU(
        input_dim=raw_sequences.shape[-1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    last_loss = float("nan")
    last_rollout_loss = float("nan")
    last_reconstruction_loss = float("nan")
    last_time_loss = float("nan")

    for _epoch in range(cfg.epochs):
        for batch, log_dt in loader:
            batch = batch.to(torch_device)
            log_dt = log_dt.to(torch_device)
            decoded_next, _z_next = model(batch)
            reconstructed, _z = model.reconstruct(batch.reshape(-1, batch.shape[-1]))
            reconstructed = reconstructed.reshape_as(batch)
            time_event = model.predict_time_event(_z_next)

            rollout_loss = torch.nn.functional.mse_loss(decoded_next, batch[:, 1:, :])
            reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, batch)
            time_loss = torch.nn.functional.mse_loss(time_event["log_dt"], log_dt)
            loss = rollout_loss + cfg.reconstruction_weight * reconstruction_loss + cfg.time_weight * time_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            last_rollout_loss = float(rollout_loss.detach().cpu())
            last_reconstruction_loss = float(reconstruction_loss.detach().cpu())
            last_time_loss = float(time_loss.detach().cpu())

    torch.save(
        {
            "net": model.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "species_names": species_names,
            "training_config": {
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "num_layers": cfg.num_layers,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "reconstruction_weight": cfg.reconstruction_weight,
                "time_weight": cfg.time_weight,
            },
            "final_metrics": {
                "loss": last_loss,
                "rollout_loss": last_rollout_loss,
                "reconstruction_loss": last_reconstruction_loss,
                "time_loss": last_time_loss,
            },
        },
        output_path,
    )

    return {
        "loss": last_loss,
        "rollout_loss": last_rollout_loss,
        "reconstruction_loss": last_reconstruction_loss,
        "time_loss": last_time_loss,
    }
