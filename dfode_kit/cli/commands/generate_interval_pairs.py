from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-interval-pairs",
        help="Convert 0D sequence trajectories into variable-dt interval-pair datasets.",
    )
    parser.add_argument("--source", required=True, help="Input sequence-v2 HDF5 file.")
    parser.add_argument("--output", required=True, help="Output interval-pair HDF5 file.")
    parser.add_argument("--min-dt", type=float, default=1e-9)
    parser.add_argument("--max-dt", type=float, default=1e-3)
    parser.add_argument("--n-dt-bins", type=int, default=12)
    parser.add_argument("--max-pairs-per-trajectory-per-bin", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--no-normalize-mass-fractions",
        action="store_false",
        dest="normalize_mass_fractions",
        default=True,
        help="Do not apply Y <- Y / sum(Y) while constructing interval pairs.",
    )


def handle_command(args):
    from dfode_kit.data.interval_pairs import IntervalPairConfig, write_interval_pair_dataset

    summary = write_interval_pair_dataset(
        args.source,
        args.output,
        config=IntervalPairConfig(
            min_dt=args.min_dt,
            max_dt=args.max_dt,
            n_dt_bins=args.n_dt_bins,
            max_pairs_per_trajectory_per_bin=args.max_pairs_per_trajectory_per_bin,
            seed=args.seed,
            normalize_mass_fractions=args.normalize_mass_fractions,
        ),
        split=args.split,
    )
    print(f"Wrote interval-pair dataset to {args.output}")
    print(
        f"n_pairs={summary['n_pairs']}, "
        f"dt_min={summary['dt_min']:.6e}, "
        f"dt_max={summary['dt_max']:.6e}"
    )
    print(f"dt_bin_counts={summary['dt_bin_counts']}")
