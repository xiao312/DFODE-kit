"""Synthetic coordinate audit; no claims about learned or chemistry accuracy."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np


def synthetic_pairs():
    """Return known increments independently of rounded endpoint subtraction."""
    pairs = []
    for initial in (0.0, 1e-30, 1e-12, 1e-6, 0.1, 0.9):
        pairs.append((initial, 0.0))
        for magnitude in np.logspace(-32, -2, 121):
            for sign in (-1.0, 1.0):
                increment = sign * magnitude
                if 0.0 <= initial + increment <= 1.0:
                    pairs.append((initial, increment))
    return np.asarray(pairs, dtype=np.float64).T


def round_trip(increments, weights, coordinate, dtype):
    if coordinate == "budget-linear":
        encoded = increments / weights
        decoded = encoded.astype(dtype).astype(np.float64) * weights
    elif coordinate == "budget-asinh":
        encoded = np.arcsinh(increments / weights)
        decoded = np.sinh(encoded.astype(dtype).astype(np.float64)) * weights
    elif coordinate == "signed-power":
        alpha = 0.1
        encoded = np.sign(increments) * np.abs(increments) ** alpha / alpha
        rounded = encoded.astype(dtype).astype(np.float64)
        decoded = np.sign(rounded) * (np.abs(rounded) * alpha) ** (1.0 / alpha)
    else:
        raise ValueError(f"Unknown coordinate: {coordinate}")
    return decoded


def summarize(reference, predicted, weights):
    error = np.abs(predicted - reference)
    active = reference != 0.0
    relative = error[active] / np.abs(reference[active])
    return {
        "count": int(reference.size),
        "nonzero_count": int(active.sum()),
        "nonzero_predicted_zero": int(np.sum(active & (predicted == 0.0))),
        "sign_mismatch_count": int(np.sum(active & (np.sign(predicted) != np.sign(reference)))),
        "relative_error_p50": float(np.quantile(relative, 0.5)) if relative.size else None,
        "relative_error_p99": float(np.quantile(relative, 0.99)) if relative.size else None,
        "relative_error_max": float(relative.max()) if relative.size else None,
        "absolute_error_max": float(error.max()),
        "budget_error_max": float(np.max(error / weights)),
        "fraction_within_budget": float(np.mean(error <= weights)),
    }


def run_audit(atol=1e-12, rtol=1e-6):
    if not np.isfinite(atol) or atol <= 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("atol must be finite and positive; rtol finite and nonnegative")
    initial, increments = synthetic_pairs()
    weights = atol + rtol * np.abs(initial)
    endpoint = initial + increments
    rows = []

    def record(name, predicted):
        row = {"name": name, **summarize(increments, predicted, weights)}
        row["magnitude_bins"] = []
        for lower, upper in ((-32, -24), (-24, -16), (-16, -8), (-8, 0)):
            selected = (np.abs(increments) >= 10.0**lower) & (np.abs(increments) < 10.0**upper)
            if selected.any():
                row["magnitude_bins"].append({
                    "log10_range": [lower, upper],
                    **summarize(increments[selected], predicted[selected], weights[selected]),
                })
        rows.append(row)

    record("endpoint-subtraction-fp64", endpoint - initial)
    record("endpoint-storage-fp32-subtract-fp64", endpoint.astype(np.float32).astype(np.float64) - initial.astype(np.float32).astype(np.float64))
    alpha = 0.1
    transform0 = (initial**alpha - 1.0) / alpha
    transform1 = (endpoint**alpha - 1.0) / alpha
    transformed_delta = transform1 - transform0
    reconstructed = np.maximum(1.0 + alpha * (transform0 + transformed_delta), 0.0) ** (1.0 / alpha)
    record("naive-boxcox-endpoint-increment-fp64", reconstructed - initial)
    for dtype in (np.float32, np.float64):
        for coordinate in ("budget-linear", "budget-asinh", "signed-power"):
            record(f"{coordinate}-encode-{np.dtype(dtype).name}-decode-fp64",
                   round_trip(increments, weights, coordinate, dtype))
    return {
        "schema_version": 1,
        "scope": "synthetic representation only; no chemistry or training",
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "budget": {"atol": atol, "rtol": rtol, "weight_state": "initial"},
        "sample_count": int(increments.size),
        "state_addition_lost_nonzero": int(np.sum((increments != 0) & (endpoint == initial))),
        "inputs": {"initial": initial.tolist(), "known_increment": increments.tolist()},
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path("runs/precision-conditioning/audit.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"sample_count": synthetic_pairs()[0].size, "output": str(args.output), "operation": "dry-run"}))
        return
    report = run_audit(args.atol, args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": report["sample_count"],
                      "state_addition_lost_nonzero": report["state_addition_lost_nonzero"]}))


if __name__ == "__main__":
    main()
