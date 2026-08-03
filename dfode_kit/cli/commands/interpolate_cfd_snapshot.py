from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "interpolate-cfd-snapshot",
        help="Augment a CFD snapshot along mesh edges on a temperature grid.",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature-spacing", type=float, required=True)
    parser.add_argument(
        "--include-original", action="store_true", default=False
    )
    parser.add_argument("--summary", default=None)


def handle_command(args):
    import json
    from pathlib import Path

    from dfode_kit.data.spatial_interpolation import (
        interpolate_temperature_grid_snapshot,
    )

    summary = interpolate_temperature_grid_snapshot(
        args.snapshot,
        args.output,
        temperature_spacing=args.temperature_spacing,
        include_original=args.include_original,
    )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
