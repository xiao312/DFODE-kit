from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "merge-interval-pairs",
        help="Merge compatible interval-pair datasets with provenance labels.",
    )
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=None)


def handle_command(args):
    import json

    from dfode_kit.data.cfd_conditioned import (
        merge_interval_pair_datasets,
    )

    summary = merge_interval_pair_datasets(
        args.source,
        args.output,
        labels=args.label,
        split=args.split,
    )
    print(json.dumps(summary, indent=2))
