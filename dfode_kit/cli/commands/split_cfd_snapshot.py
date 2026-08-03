from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "split-cfd-snapshot",
        help="Split an interpolated CFD snapshot by spatial blocks.",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--axial-blocks", type=int, default=10)
    parser.add_argument("--radial-blocks", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--mode", choices=("random", "spatial-block"), default="random"
    )
    parser.add_argument("--summary", default=None)


def handle_command(args):
    import json
    from pathlib import Path

    from dfode_kit.data.spatial_interpolation import (
        split_snapshot_random,
        split_interpolated_snapshot_spatial_blocks,
    )

    if args.mode == "random":
        summary = split_snapshot_random(
            args.snapshot,
            args.train_output,
            args.validation_output,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        summary = split_interpolated_snapshot_spatial_blocks(
            args.snapshot,
            args.train_output,
            args.validation_output,
            axial_blocks=args.axial_blocks,
            radial_blocks=args.radial_blocks,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
