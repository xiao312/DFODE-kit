#!/usr/bin/env python3
"""Deterministic distributed training for positive interval models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import cantera as ct
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel

from dfode_kit.data.interval_pairs import load_interval_pair_arrays
from dfode_kit.physics.atom_conservation import reaction_stoichiometry
from dfode_kit.training.positive_interval import (
    PositiveIntervalTrainingConfig,
    _build_model,
    _signed_power,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--final-output", default=None)
    parser.add_argument("--resume-output", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size-per-device", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--transform-alpha", type=float, default=0.1)
    parser.add_argument("--delta-loss-weight", type=float, default=0.1)
    parser.add_argument("--extent-scale", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=260624)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--non-deterministic", action="store_true")
    return parser.parse_args()


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _decode_phase(attrs: dict, override: str | None) -> str | None:
    if override:
        return override
    value = attrs.get("phase_name")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value) if value else None


def _load_core(path: str):
    loaded = load_interval_pair_arrays(path, dtype=np.float64)
    return loaded[0], loaded[1], loaded[2], loaded[5], loaded[6]


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _normalize_inputs(
    current: np.ndarray,
    dt: np.ndarray,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: np.ndarray,
    log_dt_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = ((current - state_mean) / state_std).astype(np.float32)
    log_dt = np.log(np.maximum(dt, 1.0e-300))[:, None]
    t = ((log_dt - log_dt_mean) / log_dt_std).astype(np.float32)
    return x, t


def _batch_loss(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    y0: torch.Tensor,
    target: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    alpha: float,
    delta_weight: float,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    prediction = model(x, t, current_species=y0)["next_state"]
    target_tp = (
        target[:, :2] - state_mean[None, :2]
    ) / state_std[None, :2]
    tp_loss = functional.l1_loss(prediction[:, :2], target_tp.float())
    predicted_y = prediction[:, 2:]
    target_y = target[:, 2:]
    y_loss = functional.l1_loss(
        _signed_power(predicted_y, alpha),
        _signed_power(target_y, alpha),
    )
    delta_loss = functional.l1_loss(
        _signed_power(predicted_y - y0, alpha),
        _signed_power(target_y - y0, alpha),
    )
    return (
        tp_loss + y_loss + delta_weight * delta_loss,
        (tp_loss, y_loss, delta_loss),
    )


def _make_checkpoint(
    *,
    model,
    config: PositiveIntervalTrainingConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    log_dt_mean: np.ndarray,
    log_dt_std: np.ndarray,
    gas,
    train_source: str,
    validation_source: str,
    mechanism: str,
    history: list[dict],
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    world_size: int,
    batch_size_per_device: int,
    learning_rate: float,
    minimum_learning_rate: float,
) -> dict:
    stoich = reaction_stoichiometry(gas)
    return {
        "positive_model_type": config.variant,
        "net": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "log_dt_mean": log_dt_mean,
        "log_dt_std": log_dt_std,
        "stoichiometric_matrix": stoich.net,
        "molecular_weights": np.asarray(
            gas.molecular_weights, dtype=np.float64
        ),
        "species_names": list(gas.species_names),
        "phase_name": gas.name,
        "source_path": str(train_source),
        "validation_path": str(validation_source),
        "mechanism": str(mechanism),
        "training_config": asdict(config),
        "distributed_training": {
            "world_size": world_size,
            "batch_size_per_device": batch_size_per_device,
            "global_batch_size": world_size * batch_size_per_device,
            "learning_rate": learning_rate,
            "minimum_learning_rate": minimum_learning_rate,
            "scheduler": "cosine-annealing",
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        "history": history,
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(args.num_threads)
    device = torch.device("cuda", local_rank)
    deterministic = not args.non_deterministic
    _set_reproducibility(args.seed, deterministic)

    train_current, train_target, train_dt, train_species, train_attrs = (
        _load_core(args.train_source)
    )
    val_current, val_target, val_dt, val_species, _ = _load_core(
        args.validation_source
    )
    if train_species != val_species:
        raise ValueError("Training and validation species orders differ")
    phase_name = _decode_phase(train_attrs, args.phase_name)
    gas = (
        ct.Solution(args.mechanism, phase_name)
        if phase_name
        else ct.Solution(args.mechanism)
    )
    if list(gas.species_names) != train_species:
        raise ValueError("Mechanism and dataset species orders differ")

    state_mean = train_current.mean(axis=0, dtype=np.float64)
    state_std = train_current.std(axis=0, dtype=np.float64)
    state_std = np.where(state_std > 0.0, state_std, 1.0)
    train_log_dt = np.log(np.maximum(train_dt, 1.0e-300))
    log_dt_mean = np.asarray([train_log_dt.mean()], dtype=np.float64)
    log_dt_std = np.asarray([train_log_dt.std()], dtype=np.float64)
    log_dt_std = np.where(log_dt_std > 0.0, log_dt_std, 1.0)
    train_x, train_t = _normalize_inputs(
        train_current,
        train_dt,
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
    )
    val_x, val_t = _normalize_inputs(
        val_current,
        val_dt,
        state_mean,
        state_std,
        log_dt_mean,
        log_dt_std,
    )

    config = PositiveIntervalTrainingConfig(
        variant="neural-patankar",
        epochs=args.epochs,
        batch_size=args.batch_size_per_device * world_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        transform_alpha=args.transform_alpha,
        delta_loss_weight=args.delta_loss_weight,
        extent_scale=args.extent_scale,
        seed=args.seed,
        deterministic=deterministic,
        log_every=args.log_every,
    )
    model = _build_model(config, gas, train_current.shape[1]).to(device)
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=True,
    )
    optimizer = torch.optim.AdamW(
        distributed_model.parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )

    output = Path(args.output)
    final_output = Path(args.final_output or output.with_name("final.pt"))
    resume_output = Path(args.resume_output or output.with_name("last.pt"))
    history: list[dict] = []
    start_epoch = 1
    best_epoch = 0
    best_validation_loss = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume["net"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        history = list(resume.get("history", []))
        start_epoch = int(resume["epoch"]) + 1
        best_epoch = int(resume.get("best_epoch", 0))
        best_validation_loss = float(
            resume.get("best_validation_loss", float("inf"))
        )

    state_mean_t = torch.from_numpy(state_mean).to(
        device=device, dtype=torch.float64
    )
    state_std_t = torch.from_numpy(state_std).to(
        device=device, dtype=torch.float64
    )
    usable = (train_current.shape[0] // world_size) * world_size
    local_count = usable // world_size
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "training_start",
                    "train_samples": int(train_current.shape[0]),
                    "validation_samples": int(val_current.shape[0]),
                    "usable_train_samples": int(usable),
                    "world_size": world_size,
                    "batch_size_per_device": args.batch_size_per_device,
                    "global_batch_size": (
                        args.batch_size_per_device * world_size
                    ),
                    "epochs": args.epochs,
                    "start_epoch": start_epoch,
                    "scheduler": "cosine-annealing",
                    "minimum_learning_rate": args.minimum_learning_rate,
                    "deterministic": deterministic,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, epoch])
        )
        order = epoch_rng.permutation(train_current.shape[0])[:usable]
        local_order = order[rank::world_size]
        if local_order.size != local_count:
            raise RuntimeError("Unequal distributed sample partition")

        distributed_model.train()
        local_loss_sum = torch.zeros(4, dtype=torch.float64, device=device)
        local_samples = 0
        for start in range(0, local_count, args.batch_size_per_device):
            indices = local_order[start : start + args.batch_size_per_device]
            x = torch.from_numpy(train_x[indices]).to(device)
            t = torch.from_numpy(train_t[indices]).to(device)
            y0 = torch.from_numpy(train_current[indices, 2:]).to(device)
            target = torch.from_numpy(train_target[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = _batch_loss(
                distributed_model,
                x,
                t,
                y0,
                target,
                state_mean_t,
                state_std_t,
                args.transform_alpha,
                args.delta_loss_weight,
            )
            loss.backward()
            optimizer.step()
            count = indices.size
            local_loss_sum[0] += loss.detach().double() * count
            for component_index, component in enumerate(components, start=1):
                local_loss_sum[component_index] += (
                    component.detach().double() * count
                )
            local_samples += count

        scheduler.step()
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        sample_count = torch.tensor(
            [local_samples], dtype=torch.float64, device=device
        )
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
        train_losses = (local_loss_sum / sample_count[0]).cpu().numpy()

        should_validate = (
            epoch == 1
            or epoch % args.log_every == 0
            or epoch == args.epochs
        )
        validation_loss = torch.tensor(
            [float("nan")], dtype=torch.float64, device=device
        )
        if should_validate:
            dist.barrier()
            if rank == 0:
                model.eval()
                total = 0.0
                count = 0
                with torch.inference_mode():
                    for start in range(
                        0, val_current.shape[0], args.batch_size_per_device
                    ):
                        stop = min(
                            start + args.batch_size_per_device,
                            val_current.shape[0],
                        )
                        x = torch.from_numpy(val_x[start:stop]).to(device)
                        t = torch.from_numpy(val_t[start:stop]).to(device)
                        y0 = torch.from_numpy(
                            val_current[start:stop, 2:]
                        ).to(device)
                        target = torch.from_numpy(
                            val_target[start:stop]
                        ).to(device)
                        loss, _ = _batch_loss(
                            model,
                            x,
                            t,
                            y0,
                            target,
                            state_mean_t,
                            state_std_t,
                            args.transform_alpha,
                            args.delta_loss_weight,
                        )
                        batch_count = stop - start
                        total += float(loss) * batch_count
                        count += batch_count
                validation_loss[0] = total / count
            dist.broadcast(validation_loss, src=0)

        if rank == 0 and should_validate:
            value = float(validation_loss[0])
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": float(train_losses[0]),
                "train_tp_loss": float(train_losses[1]),
                "train_y_loss": float(train_losses[2]),
                "train_delta_loss": float(train_losses[3]),
                "validation_loss": value,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            if value < best_validation_loss:
                best_validation_loss = value
                best_epoch = epoch
                checkpoint = _make_checkpoint(
                    model=model,
                    config=config,
                    state_mean=state_mean,
                    state_std=state_std,
                    log_dt_mean=log_dt_mean,
                    log_dt_std=log_dt_std,
                    gas=gas,
                    train_source=args.train_source,
                    validation_source=args.validation_source,
                    mechanism=args.mechanism,
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    world_size=world_size,
                    batch_size_per_device=args.batch_size_per_device,
                    learning_rate=args.learning_rate,
                    minimum_learning_rate=args.minimum_learning_rate,
                )
                _atomic_save(checkpoint, output)
                record["saved_best"] = True
            print(json.dumps(record, sort_keys=True), flush=True)

        should_checkpoint = (
            epoch % args.checkpoint_every == 0
            or epoch == args.epochs
        )
        if rank == 0 and should_checkpoint:
            resume_payload = _make_checkpoint(
                model=model,
                config=config,
                state_mean=state_mean,
                state_std=state_std,
                log_dt_mean=log_dt_mean,
                log_dt_std=log_dt_std,
                gas=gas,
                train_source=args.train_source,
                validation_source=args.validation_source,
                mechanism=args.mechanism,
                history=history,
                epoch=epoch,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                world_size=world_size,
                batch_size_per_device=args.batch_size_per_device,
                learning_rate=args.learning_rate,
                minimum_learning_rate=args.minimum_learning_rate,
            )
            resume_payload.update(
                {
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
            )
            _atomic_save(resume_payload, resume_output)

    if rank == 0:
        final_checkpoint = _make_checkpoint(
            model=model,
            config=config,
            state_mean=state_mean,
            state_std=state_std,
            log_dt_mean=log_dt_mean,
            log_dt_std=log_dt_std,
            gas=gas,
            train_source=args.train_source,
            validation_source=args.validation_source,
            mechanism=args.mechanism,
            history=history,
            epoch=args.epochs,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            world_size=world_size,
            batch_size_per_device=args.batch_size_per_device,
            learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
        )
        _atomic_save(final_checkpoint, final_output)
        summary = {
            "event": "training_complete",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint": str(output),
            "final_checkpoint": str(final_output),
            "resume_checkpoint": str(resume_output),
            "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
