from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-cfd-conditioned-sequences",
        help="Generate constant-pressure Cantera trajectories from CFD cell states.",
    )
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--mech", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--mechanism-id", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--min-time", type=float, default=1e-9)
    parser.add_argument("--max-time", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--energy", choices=("on", "off"), default="on")
    parser.add_argument("--summary", default=None)


def handle_command(args):
    import json
    from pathlib import Path

    from dfode_kit.data.cfd_conditioned import (
        generate_cfd_conditioned_sequences,
    )

    summary = generate_cfd_conditioned_sequences(
        args.snapshot,
        args.mech,
        args.train_output,
        args.validation_output,
        phase_name=args.phase_name,
        mechanism_id=args.mechanism_id,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_time=args.min_time,
        max_time=args.max_time,
        steps=args.steps,
        energy=args.energy,
    )
    if args.summary:
        Path(args.summary).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2))
