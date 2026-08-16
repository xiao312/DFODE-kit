#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dfode_kit.data.interval_pairs import load_interval_pair_arrays
from dfode_kit.evaluation.ood import envelope_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quantile", type=float, default=0.999)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    states, _target, dt, _bins, _edges, species, attrs = (
        load_interval_pair_arrays(args.dataset, dtype=np.float64)
    )
    envelope, summary = envelope_from_checkpoint(
        checkpoint, states, dt, quantile=args.quantile
    )
    payload = envelope.to_dict()
    payload.update(
        {
            "summary": summary,
            "species_names": species,
            "label_backend": str(attrs.get("label_backend", "")),
            "source_dataset": str(Path(args.dataset).resolve()),
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
