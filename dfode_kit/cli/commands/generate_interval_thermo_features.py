from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-interval-thermo-features",
        help="Precompute chemical-potential affinity features for interval-pair datasets.",
    )
    parser.add_argument("--source", required=True, help="Input interval-pair HDF5 file to update in place.")
    parser.add_argument("--mech", required=True, help="Cantera mechanism file.")
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--mole-fraction-floor", type=float, default=1e-300)


def handle_command(args):
    from dfode_kit.data.interval_pairs import add_interval_thermo_features

    summary = add_interval_thermo_features(
        args.source,
        args.mech,
        phase_name=args.phase_name,
        chunk_size=args.chunk_size,
        mole_fraction_floor=args.mole_fraction_floor,
    )
    print(f"Wrote thermo interval features into {args.source}")
    print(
        f"n_pairs={summary['n_pairs']}, "
        f"n_reactions={summary['n_reactions']}, "
        f"phase_name={summary['phase_name']}"
    )
