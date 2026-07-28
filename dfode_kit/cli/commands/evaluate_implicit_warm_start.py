from __future__ import annotations


def add_command_parser(subparsers):
    parser = subparsers.add_parser(
        "evaluate-implicit-warm-start",
        help="Evaluate neural guesses for a damped-Newton Backward-Euler chemistry solve.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trained stoichiometric interval checkpoint.")
    parser.add_argument("--source", required=True, help="Interval-pair HDF5 evaluation dataset.")
    parser.add_argument("--mechanism", required=True, help="Cantera YAML mechanism path or built-in name.")
    parser.add_argument("--phase-name", help="Optional Cantera phase name; defaults to dataset metadata.")
    parser.add_argument("--output", required=True, help="JSON report path; a sibling sample CSV is also written.")
    parser.add_argument("--device", help="Torch device for neural inference, for example cuda:0 or cpu.")
    parser.add_argument("--max-samples", type=int, default=32, help="Maximum stratified interval pairs; <=0 uses all.")
    parser.add_argument("--sample-seed", type=int, default=20260713)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--species-absolute-tolerance", type=float, default=1e-12)
    parser.add_argument("--temperature-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--newton-tolerance", type=float, default=1.0, help="Converged WRMS residual threshold.")
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--max-line-search-steps", type=int, default=12)
    parser.add_argument("--fd-relative-step", type=float, default=1e-5)
    parser.add_argument("--fd-species-scale", type=float, default=1e-5)
    parser.add_argument("--minimum-temperature", type=float, default=200.0)
    parser.add_argument("--positivity-floor", type=float, default=0.0)
    parser.add_argument("--skip-cvode", action="store_true", help="Skip direct ReactorNet timing and statistics.")
    parser.set_defaults(command_handler=handle_command)


def handle_command(args):
    from dfode_kit.evaluation.implicit_warm_start import evaluate_implicit_warm_start

    result = evaluate_implicit_warm_start(
        args.checkpoint,
        args.source,
        args.mechanism,
        args.output,
        phase_name=args.phase_name,
        device=args.device,
        max_samples=args.max_samples,
        sample_seed=args.sample_seed,
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
        run_cvode=not args.skip_cvode,
    )

    print(f"wrote implicit warm-start report: {args.output}")
    print(f"wrote per-sample report: {result['sample_csv']}")
    for name, metrics in result["guess_metrics"].items():
        residual = metrics["initial_wrms"]["median"]
        iterations = metrics["iterations_when_converged"]["mean"]
        print(
            f"{name}: valid={metrics['initial_valid_rate']:.3f}, "
            f"converged={metrics['convergence_rate']:.3f}, "
            f"median_initial_wrms={residual}, mean_converged_iterations={iterations}"
        )
    runtime = result["runtime_comparison"]
    if "cvode_advance_microseconds_per_sample" in runtime:
        print(f"CVODE advance: {runtime['cvode_advance_microseconds_per_sample']:.3f} us/sample")
