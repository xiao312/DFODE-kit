#!/usr/bin/env python3
"""Evaluate an exported Fluent artifact on untouched CFD interval pairs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def _decode(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _safe_cosine(predicted: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(predicted.ravel(), target.ravel()) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reactive-threshold", type=float, default=1.0e-8)
    parser.add_argument("--sign-threshold", type=float, default=1.0e-12)
    parser.add_argument("--max-normalized-mae", type=float, default=1.0)
    parser.add_argument("--minimum-global-cosine", type=float, default=0.5)
    parser.add_argument("--positivity-tolerance", type=float, default=1.0e-15)
    args = parser.parse_args()

    with h5py.File(args.source, "r") as handle:
        current = np.asarray(handle["pairs/current_states"], dtype=np.float64)
        target = np.asarray(handle["pairs/target_states"], dtype=np.float64)
        dt = np.asarray(handle["pairs/dt"], dtype=np.float64)
        species_names = _decode(np.asarray(handle["species_names"]))

    model = torch.jit.load(
        str(args.artifact_dir / "model.pt"), map_location=args.device
    )
    model.eval()
    manifest = json.loads(
        (args.artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    separate_species_input = (
        manifest.get("hard_layer", {}).get("current_species_source")
        == "separate_input"
    )
    predicted_delta_parts = []
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, current.shape[0], args.batch_size):
            end = min(begin + args.batch_size, current.shape[0])
            model_input = np.concatenate(
                [current[begin:end], dt[begin:end, None]], axis=1
            ).astype(np.float32, copy=False)
            model_tensor = torch.from_numpy(model_input).to(args.device)
            if separate_species_input:
                species_tensor = torch.from_numpy(
                    np.ascontiguousarray(
                        current[begin:end, 2:],
                        dtype=np.float64,
                    )
                ).to(args.device)
                result = model(model_tensor, species_tensor)
            else:
                result = model(model_tensor)
            if isinstance(result, (tuple, list)):
                result = result[0]
            predicted_delta_parts.append(
                result.detach().cpu().numpy().astype(np.float64, copy=False)
            )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    predicted_delta = np.concatenate(predicted_delta_parts, axis=0)
    true_delta = target[:, 2:] - current[:, 2:]
    predicted_target = current[:, 2:] + predicted_delta
    current_species = current[:, 2:]
    target_species = target[:, 2:]
    error = predicted_delta - true_delta
    true_activity = np.max(np.abs(true_delta), axis=1)
    reactive = true_activity > args.reactive_threshold

    row_pred_norm = np.linalg.norm(predicted_delta, axis=1)
    row_true_norm = np.linalg.norm(true_delta, axis=1)
    row_denominator = row_pred_norm * row_true_norm
    valid_cosine = reactive & (row_denominator > 0.0)
    row_cosine = np.full(current.shape[0], np.nan, dtype=np.float64)
    row_cosine[valid_cosine] = np.sum(
        predicted_delta[valid_cosine] * true_delta[valid_cosine], axis=1
    ) / row_denominator[valid_cosine]

    absolute_error_sum = float(np.sum(np.abs(error)))
    true_absolute_sum = float(np.sum(np.abs(true_delta)))
    normalized_mae = absolute_error_sum / max(true_absolute_sum, 1.0e-300)
    global_cosine = _safe_cosine(predicted_delta[reactive], true_delta[reactive])
    sign_mask = np.abs(true_delta) > args.sign_threshold
    sign_agreement = float(
        np.mean(
            np.sign(predicted_delta[sign_mask])
            == np.sign(true_delta[sign_mask])
        )
    )

    per_species = []
    for index, species in enumerate(species_names):
        species_true = true_delta[:, index]
        species_error = error[:, index]
        denominator = float(np.sum(np.abs(species_true)))
        per_species.append(
            {
                "species": species,
                "delta_mae": float(np.mean(np.abs(species_error))),
                "true_delta_mean_abs": float(np.mean(np.abs(species_true))),
                "normalized_absolute_error": float(
                    np.sum(np.abs(species_error)) / max(denominator, 1.0e-300)
                ),
                "cosine": _safe_cosine(
                    predicted_delta[:, index], species_true
                ),
            }
        )
    per_species.sort(
        key=lambda item: item["delta_mae"], reverse=True
    )

    gate_pass = (
        normalized_mae <= args.max_normalized_mae
        and np.isfinite(global_cosine)
        and global_cosine >= args.minimum_global_cosine
        and not np.any(predicted_target < -args.positivity_tolerance)
    )
    report = {
        "artifact_dir": str(args.artifact_dir),
        "source": str(args.source),
        "n_pairs": int(current.shape[0]),
        "n_species": len(species_names),
        "reactive_threshold": args.reactive_threshold,
        "reactive_pair_fraction": float(np.mean(reactive)),
        "delta_y": {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "true_mean_abs": float(np.mean(np.abs(true_delta))),
            "predicted_mean_abs": float(np.mean(np.abs(predicted_delta))),
            "normalized_mae": normalized_mae,
            "global_reactive_cosine": global_cosine,
            "median_reactive_row_cosine": float(
                np.nanmedian(row_cosine[reactive])
            ),
            "p10_reactive_row_cosine": float(
                np.nanpercentile(row_cosine[reactive], 10)
            ),
            "sign_agreement": sign_agreement,
        },
        "composition": {
            "negative_entry_rate": float(np.mean(predicted_target < 0.0)),
            "negative_row_rate": float(
                np.mean(np.any(predicted_target < 0.0, axis=1))
            ),
            "negative_below_tolerance_entry_rate": float(
                np.mean(predicted_target < -args.positivity_tolerance)
            ),
            "negative_below_tolerance_row_rate": float(
                np.mean(
                    np.any(
                        predicted_target < -args.positivity_tolerance,
                        axis=1,
                    )
                )
            ),
            "minimum_mass_fraction": float(np.min(predicted_target)),
            "positivity_tolerance": args.positivity_tolerance,
            "input_minimum_mass_fraction": float(
                np.min(current_species)
            ),
            "input_negative_entry_rate": float(
                np.mean(current_species < 0.0)
            ),
            "target_minimum_mass_fraction": float(
                np.min(target_species)
            ),
            "target_negative_entry_rate": float(
                np.mean(target_species < 0.0)
            ),
            "mean_mass_sum_error": float(
                np.mean(np.abs(np.sum(predicted_target, axis=1) - 1.0))
            ),
            "max_mass_sum_error": float(
                np.max(np.abs(np.sum(predicted_target, axis=1) - 1.0))
            ),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "microseconds_per_pair": 1.0e6 * elapsed / current.shape[0],
        },
        "gate": {
            "pass": bool(gate_pass),
            "max_normalized_mae": args.max_normalized_mae,
            "minimum_global_cosine": args.minimum_global_cosine,
            "positivity_tolerance": args.positivity_tolerance,
            "requires_nonnegative_prediction": True,
        },
        "per_species_by_delta_mae": per_species,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
