"""Convert an exact inspected checkpoint using a patched, restricted loader."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HASHES = {
    "state-boxcox": "7938fc47584da9cad2b811e9607ed6413bf210d3ae4c20f3dca64807ae56abee",
    "signed-power": "8ed4105be6eab440b57141072dc384a4df2454d7570df5a7647a10599a471089",
}
MECHANISM = "26a27fb3c19c6000ed46d70947faeaf4813b6161ca7186fb6cc9ad55ede294f0"
SHAPES = [(800, 61), (800, 800), (800, 800), (800, 800), (58, 800)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_array(value, shape, name):
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError(f"Unsupported tensor storage: {name}")
        value = value.detach().numpy()
    elif not isinstance(value, np.ndarray):
        raise ValueError(f"Expected a numerical tensor or array: {name}")
    if value.dtype not in (np.dtype("float32"), np.dtype("float64")) or value.shape != shape:
        raise ValueError(f"Unexpected dtype/shape for {name}: {value.dtype}/{value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite checkpoint data: {name}")
    return np.array(value, copy=True)


def checked_arrays(payload, kind):
    if type(payload) is not dict:
        raise ValueError("Expected the inspected dictionary payload")
    if kind == "signed-power":
        if set(payload) != {"model_state_dict", "normalization_stats"}:
            raise ValueError("Unexpected direct-power checkpoint keys")
        weights, stats = payload["model_state_dict"], payload["normalization_stats"]
    else:
        if set(payload) != {"net", "data_in_mean", "data_in_std", "data_target_mean", "data_target_std"}:
            raise ValueError("Unexpected conventional checkpoint keys")
        weights = payload["net"]
        stats = {"features_mean": payload["data_in_mean"], "features_std": payload["data_in_std"],
                 "labels_mean": payload["data_target_mean"], "labels_std": payload["data_target_std"]}
    expected = {f"net.linear_layer_{index}.{suffix}" for index in range(5) for suffix in ("weight", "bias")}
    if set(weights) != expected or set(stats) != {"features_mean", "features_std", "labels_mean", "labels_std"}:
        raise ValueError("Checkpoint parameter names differ from the inspected source")
    arrays = {}
    for index, shape in enumerate(SHAPES):
        for suffix, size in (("weight", shape), ("bias", (shape[0],))):
            name = f"layer{index}_{suffix}"
            arrays[name] = numeric_array(weights[f"net.linear_layer_{index}.{suffix}"], size, name)
            if arrays[name].dtype != np.float32:
                raise ValueError("Historical network weights must be FP32")
    for name, size in (("features_mean", 61), ("features_std", 61), ("labels_mean", 58), ("labels_std", 58)):
        arrays[name] = numeric_array(stats[name], (size,), name)
        if name.endswith("_std") and np.any(arrays[name] <= 0):
            raise ValueError(f"Nonpositive scale in {name}; do not invent a normalization repair")
    if kind == "signed-power" and np.any(arrays["labels_mean"] != 0):
        raise ValueError("Expected the original zero direct-power target mean")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=HASHES, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size > 9_000_000:
        raise ValueError("Checkpoint exceeds the inspected size boundary")
    if digest(args.checkpoint) != HASHES[args.kind]:
        raise ValueError("Checkpoint is not one of the two inspected, preselected artifacts")
    contract = json.loads(args.contract.read_text())
    species = contract["species_names"]
    if contract["mechanism_sha256"] != MECHANISM or len(species) != 59 or species[-1] != "AR":
        raise ValueError("External species/mechanism contract differs")
    plan = {"kind": args.kind, "checkpoint_sha256": HASHES[args.kind], "checkpoint_bytes": args.checkpoint.stat().st_size,
            "torch": torch.__version__, "numpy": np.__version__, "weights_only": True,
            "mechanism_sha256": MECHANISM, "species_names": species, "interval_s": 1e-6,
            "architecture": [61, 800, 800, 800, 800, 58], "activation": "GELU", "lambda": .1,
            "input_units": ["K", "Pa", "mechanism-ordered mass fractions"],
            "source_contract_sha256": digest(args.contract), "converter_sha256": digest(__file__),
            "qualification": "External mechanism/species contract; historical provenance is not a verified paper checkpoint",
            "normalization_limit": "Power statistics retain FP32 only; exact original FP64 preprocessing cannot be recovered"}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    if torch.__version__ != "2.10.0+cpu" or np.__version__ != "2.2.6":
        raise ValueError("Require the separate pinned Torch 2.10.0+cpu / NumPy 2.2.6 converter environment")
    torch.set_num_threads(1)
    # These NumPy symbols reconstruct the inspected FP64 normalization arrays.
    # No user-defined class or function is allowed; there is no unsafe fallback.
    allowed = [] if args.kind == "signed-power" else [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    arrays = checked_arrays(payload, args.kind)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "model.npz", **arrays)
    with np.load(args.output / "model.npz", allow_pickle=False) as replay:
        for name, value in arrays.items():
            np.testing.assert_array_equal(replay[name], value)
    plan.update(status="converted", arrays_sha256=digest(args.output / "model.npz"),
                arrays={name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()})
    (args.output / "manifest.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    print(json.dumps({"status": "converted", "kind": args.kind, "arrays": len(arrays), "sha256": plan["arrays_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
