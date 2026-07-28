from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "generate-implicit-root-data",
        help="Generate Backward-Euler root pairs using adaptive continuation and exact Cantera residuals.",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanism", required=True)
    parser.add_argument("--phase-name")
    parser.add_argument("--max-source-samples", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=20260713)
    parser.add_argument("--max-accepted-steps", type=int, default=128)
    parser.add_argument("--min-substep", type=float, default=1e-12)
    parser.add_argument("--n-dt-bins", type=int, default=12)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--species-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--temperature-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--newton-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-newton-iterations", type=int, default=16)
    parser.add_argument("--max-line-search-steps", type=int, default=16)
    parser.add_argument("--fd-relative-step", type=float, default=1e-5)
    parser.add_argument("--fd-species-scale", type=float, default=1e-5)
    parser.add_argument("--minimum-temperature", type=float, default=200.0)
    parser.add_argument("--positivity-floor", type=float, default=0.0)
    parser.set_defaults(command_handler=handle_command)


def handle_command(args):
    from dfode_kit.data.implicit_roots import generate_implicit_root_dataset

    result = generate_implicit_root_dataset(
        args.source,
        args.output,
        args.mechanism,
        phase_name=args.phase_name,
        max_source_samples=args.max_source_samples,
        sample_seed=args.sample_seed,
        max_accepted_steps=args.max_accepted_steps,
        min_substep=args.min_substep,
        n_dt_bins=args.n_dt_bins,
        relative_tolerance=args.relative_tolerance,
        species_absolute_tolerance=args.species_absolute_tolerance,
        temperature_absolute_tolerance=args.temperature_absolute_tolerance,
        newton_tolerance=args.newton_tolerance,
        max_newton_iterations=args.max_newton_iterations,
        max_line_search_steps=args.max_line_search_steps,
        fd_relative_step=args.fd_relative_step,
        fd_species_scale=args.fd_species_scale,
        minimum_temperature=args.minimum_temperature,
        positivity_floor=args.positivity_floor,
    )
    print(f"wrote implicit-root dataset: {args.output}")
    print(f"wrote generation report: {result['summary_path']}")
    print(
        f"intervals={result['n_selected_intervals']} completed={result['completion_rate']:.3f} "
        f"root_pairs={result['n_root_pairs']} dt=[{result['root_dt_min']:.3e}, {result['root_dt_max']:.3e}]"
    )

