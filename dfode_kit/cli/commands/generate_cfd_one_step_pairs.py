from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-cfd-one-step-pairs",
        help="Generate fixed-dt Cantera labels for a CFD snapshot.",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--mech", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--energy", choices=("on", "off"), default="on")
    parser.add_argument("--summary", default=None)


def handle_command(args):
    import json
    from pathlib import Path

    from dfode_kit.data.cfd_one_step import generate_cfd_one_step_pairs

    summary = generate_cfd_one_step_pairs(
        args.snapshot,
        args.mech,
        args.output,
        dt=args.dt,
        phase_name=args.phase_name,
        split=args.split,
        energy=args.energy,
    )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
