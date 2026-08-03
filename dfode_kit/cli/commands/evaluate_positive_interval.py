from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-positive-interval",
        help="Evaluate positive interval models or positivity projections.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mech", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=("none", "sandu-composition", "sandu-reaction", "relative"),
        default="none",
    )
    parser.add_argument("--floor", type=float, default=1e-30)
    parser.add_argument("--device")
    return parser


def handle_command(args):
    from dfode_kit.evaluation.positive_interval import evaluate_positive_interval

    report = evaluate_positive_interval(
        args.checkpoint,
        args.source,
        args.mech,
        args.output,
        method=args.method,
        device=args.device,
        floor=args.floor,
    )
    print(report)
